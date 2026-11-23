#!/usr/bin/env python3
"""Spectrum Predictor (ms-pred) bridge script.

Runs INSIDE the ms-pred repo's own virtualenv
(``~/ms-pred/.venv/bin/python``), never inside the GUI's ``deepmass2``
conda env -- ms-pred's dependency stack (torch, pytorch_lightning, dgl-style
geometric ops, ``multiprocess``, ``h5py``, ``seaborn``, ...) is isolated
there on purpose, mirroring how the previous MassFormer adapter kept its
model environment separate from the GUI process.

Two subcommands, each reading a JSON request file and writing a JSON
response file (avoids stdin/stdout encoding edge cases across the
subprocess boundary):

  predict --input req.json --output resp.json
      Predict MS/MS spectra for a batch of candidate SMILES with Spectrum Predictor.
      Emits, per candidate id: m/z array, intensity array, per-peak
      fragment id (an integer atom-inclusion bitmask -- see
      ``ms_pred.magma.fragmentation.FragmentEngine.get_draw_dict``), the
      parent molecule's atom count, and the exact (round-trip-canonical)
      SMILES string Spectrum Predictor used internally, which callers must reuse
      verbatim for ``render`` so atom indices line up.

  render --input req.json --output resp.json
      Render an RDKit SVG highlighting the substructure a given fragment id
      corresponds to, for a given (already-canonical) SMILES. Ports the
      ``/fragment_svg`` Flask route in ``ms-pred/webui/app.py`` to a
      standalone script. No model loading -- fast.
"""
import argparse
import json
import sys
from pathlib import Path


def _write_error(output_path, msg):
    Path(output_path).write_text(json.dumps({"error": str(msg)}))


def cmd_predict(mspred_dir: Path, args):
    payload = json.loads(Path(args.input).read_text())
    candidates = payload["candidates"]  # [{"id": str, "smiles": str}, ...]
    adduct = payload["adduct"]
    nce = float(payload.get("nce", 40.0))
    instrument = str(payload.get("instrument") or "Orbitrap")
    gpu = bool(payload.get("gpu", False))
    gen_ckpt = payload.get("gen_checkpoint", "models/msg_all/gen/best.ckpt")
    inten_ckpt = payload.get("inten_checkpoint", "models/msg_all/inten_contr/best.ckpt")
    sparse_k = int(payload.get("sparse_k", 100))
    timeout = float(payload.get("timeout", 600))

    import pandas as pd
    from rdkit import Chem

    sys.path.insert(0, str(mspred_dir / "src"))
    import ms_pred.common as common

    if adduct not in common.ion2mass:
        _write_error(args.output, f"Unsupported adduct for Spectrum Predictor: {adduct}")
        return

    rows = []
    id_order = []
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand["smiles"])
        if mol is None:
            continue
        # Match predict_smis.py's own canonicalization exactly (roundtrip
        # via Chem.MolToSmiles(Chem.MolFromSmiles(...))) so the SMILES we
        # cache for later fragment rendering produces the identical atom
        # ordering FragmentEngine used when the bitmask was computed.
        canon_smi = Chem.MolToSmiles(mol)
        formula = common.form_from_smi(canon_smi)
        precursor = common.formula_mass(formula) + common.ion2mass[adduct]
        cand_id = str(cand["id"])
        rows.append({
            "smiles": canon_smi,
            "ionization": adduct,
            "precursor": precursor,
            "instrument": instrument,
            "spec": cand_id,
            "collision_energies": str([nce]),
        })
        id_order.append(cand_id)

    results = {}
    if not rows:
        Path(args.output).write_text(json.dumps({"results": results}))
        return

    import tempfile
    import subprocess

    # ms-pred is a model/code installation and may be read-only (as it is in
    # packaged/headless deployments). Prediction intermediates belong in the
    # OS temporary area, not inside the ms-pred checkout.
    with tempfile.TemporaryDirectory(prefix="annomass_iceberg_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        labels_fp = tmpdir / "labels.tsv"
        pd.DataFrame(rows).to_csv(labels_fp, sep="\t", index=False)

        out_h5_name = "preds.hdf5"
        cmd = [
            sys.executable,
            "src/ms_pred/iceberg/predict_smis.py",
            "--gen-checkpoint", gen_ckpt,
            "--inten-checkpoint", inten_ckpt,
            "--dataset-labels", str(labels_fp),
            "--sparse-out",
            "--sparse-k", str(sparse_k),
            "--save-dir", str(tmpdir),
            "--out-name", out_h5_name,
            "--adduct-shift",
        ]
        if gpu:
            cmd.append("--gpu")

        proc = subprocess.run(
            cmd, cwd=str(mspred_dir), capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            _write_error(args.output, f"predict_smis.py failed:\n{proc.stderr[-2000:]}")
            return

        h5_path = tmpdir / out_h5_name
        if not h5_path.is_file():
            _write_error(args.output, "predict_smis.py did not produce an output file")
            return

        specdb = common.PredSpecDB(h5_path=h5_path, mode="r")
        row_by_id = {r["spec"]: r for r in rows}
        for cand_id in id_order:
            name = f"pred_{cand_id}"
            try:
                spec_dict, has_remark = specdb.read_from_name(name)
            except Exception as e:
                results[cand_id] = {"error": str(e)}
                continue
            if not spec_dict:
                results[cand_id] = {"error": "no prediction returned"}
                continue
            # predict_smis.py always tags entries with a remark (the
            # candidate's InChIKey), so read_from_name nests one level
            # deeper: {remark: {"collision X": MassSpec}}. We only ever
            # request a single remark and a single collision energy, so
            # take the first (only) entry at each level.
            inner = next(iter(spec_dict.values()))
            spec = next(iter(inner.values())) if has_remark else inner
            # Different (fragment, H-shift) combinations frequently land on
            # the same nominal mass in the raw sparse output; merge them
            # with ms-pred's own MassSpec.merged_spec() (sum method) rather
            # than a hand-rolled equivalent, so this matches whatever the
            # reference tooling built on top of ms-pred produces exactly,
            # by construction, not just numerically. It groups by (rounded
            # m/z, exact fragment id) so it never conflates two genuinely
            # different fragments that happen to share a nominal mass.
            merged = spec.merged_spec(merge_method="sum")
            # Atom count of the parent molecule == bit width of each
            # fragment id; derive it from the canonical SMILES we sent.
            frag_bits = Chem.MolFromSmiles(row_by_id[cand_id]["smiles"]).GetNumAtoms()
            results[cand_id] = {
                "mz": [float(x) for x in merged.masses],
                "intensity": [float(x) for x in merged.intens],
                "frag_ids": [int(x) for x in merged.int_frags] if merged.has_frags else [],
                "frag_bits": int(frag_bits),
                "canonical_smiles": row_by_id[cand_id]["smiles"],
            }

    Path(args.output).write_text(json.dumps({"results": results}))


def cmd_render(mspred_dir: Path, args):
    payload = json.loads(Path(args.input).read_text())
    smiles = (payload.get("smiles") or "").strip()
    frag_id = payload.get("frag_id")
    mz_obs = payload.get("mz")
    size_px = int(payload.get("size", 320))
    size_px = max(120, min(800, size_px))

    if not smiles or frag_id is None:
        _write_error(args.output, "Missing 'smiles' or 'frag_id'")
        return

    sys.path.insert(0, str(mspred_dir / "src"))
    import ms_pred.magma.fragmentation as fragmentation
    from rdkit.Chem.Draw import rdMolDraw2D

    try:
        frag_id_int = int(frag_id)
        # ms-pred's own prediction path always canonicalizes SMILES before
        # building the FragmentEngine that produced this bitmask, so we
        # must reconstruct with the identical setting to keep atom indices
        # aligned -- see docstring at the top of this file.
        engine = fragmentation.FragmentEngine(smiles, mol_str_type="smiles", mol_str_canonicalized=True)
        draw_dict = engine.get_draw_dict(frag_id_int)

        mol = draw_dict["mol"]
        hatoms = list(draw_dict.get("hatoms") or [])
        hbonds = list(draw_dict.get("hbonds") or [])

        d2d = rdMolDraw2D.MolDraw2DSVG(size_px, size_px)
        d2d.DrawMolecule(mol, highlightAtoms=hatoms, highlightBonds=hbonds)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()

        h_shift = 0
        if mz_obs is not None:
            try:
                base_mass = float(engine.single_mass(frag_id_int))
                h_mass = 1.00784
                h_shift = int(round((float(mz_obs) - base_mass) / h_mass))
            except Exception:
                h_shift = 0
        try:
            formula = engine.formula_from_frag(frag_id_int, h_shift=h_shift)
        except Exception:
            formula = ""

        Path(args.output).write_text(json.dumps({
            "svg": svg,
            "formula": formula,
            "h_shift": h_shift,
        }))
    except Exception as e:
        _write_error(args.output, f"Failed to draw fragment: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["predict", "render"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mspred-dir", default=str(Path.home() / "ms-pred"))
    args = parser.parse_args()

    mspred_dir = Path(args.mspred_dir).expanduser().resolve()

    try:
        if args.command == "predict":
            cmd_predict(mspred_dir, args)
        else:
            cmd_render(mspred_dir, args)
    except Exception as e:
        _write_error(args.output, f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
