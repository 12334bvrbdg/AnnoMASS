"""Automatic ICEBERG figure export for GUI and headless AnnoMASS runs."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors


def _peaks(spectrum):
    try:
        return np.asarray(spectrum.peaks.mz, float), np.asarray(spectrum.peaks.intensities, float)
    except Exception:
        return np.asarray(spectrum.get("mz") or [], float), np.asarray(spectrum.get("intensity") or [], float)


def _safe_name(value, max_len=60):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "candidate"))
    value = value.strip("._") or "candidate"
    return value[:max_len].rstrip("._") or "candidate"


def _candidate_id(row, ordinal):
    for col in ("confidence_candidate_id", "candidate_id"):
        value = row.get(col)
        if value is not None and str(value).strip() not in ("", "nan"):
            return _safe_name(value)
    name = str(row.get("compound_name", "") or "").strip()
    if name and name.lower() not in ("nan", "unknown", "none"):
        # Ordinal prefix keeps filenames unique (and stably sorted) even when
        # the same compound is the top hit for more than one spectrum.
        return f"{ordinal:02d}_{_safe_name(name)}"
    try:
        return f"HC{int(row.get('deepmet_rank')):02d}"
    except Exception:
        return f"HC{ordinal:02d}"


def _cache_for(row, ordinal, full_df, cache):
    try:
        key = int(row.get("_mf_cache_row"))
        if key in cache:
            return cache[key]
    except Exception:
        pass
    if full_df is not None and len(full_df):
        mask = np.ones(len(full_df), dtype=bool)
        if "smiles" in full_df and str(row.get("smiles", "")):
            mask &= full_df["smiles"].astype(str).eq(str(row.get("smiles"))).to_numpy()
        if "spectrum_index" in full_df:
            try:
                mask &= full_df["spectrum_index"].astype(int).eq(int(row.get("spectrum_index"))).to_numpy()
            except Exception:
                pass
        hits = np.flatnonzero(mask)
        if len(hits) and int(hits[0]) in cache:
            return cache[int(hits[0])]
    return cache.get(ordinal - 1)


def _mirror(candidate_id, row, spectrum, pred, out_dir):
    emz, eint = _peaks(spectrum)
    pmz = np.asarray(pred.get("mz", []), float); pint = np.asarray(pred.get("intensity", []), float)
    if not len(emz) or not len(pmz):
        return []
    eint = 100 * eint / max(float(eint.max()), 1e-12)
    pint = 100 * pint / max(float(pint.max()), 1e-12)
    matched_e, matched_p = set(), set()
    for j, mass in enumerate(pmz):
        i = int(np.argmin(np.abs(emz - mass)))
        if abs(float(emz[i] - mass)) <= max(0.01, float(mass) * 10e-6):
            matched_e.add(i); matched_p.add(j)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, (mass, inten) in enumerate(zip(emz, eint)):
        ax.vlines(mass, 0, inten, color="#159447" if i in matched_e else "#d73027", lw=1.15)
    for j, (mass, inten) in enumerate(zip(pmz, pint)):
        ax.vlines(mass, 0, -inten, color="#159447" if j in matched_p else "#2166ac", lw=1.15)
    ax.axhline(0, color="#222", lw=.7); ax.set_ylim(-112, 112)
    ax.set_yticks([-100, -50, 0, 50, 100]); ax.set_yticklabels([100, 50, 0, 50, 100])
    ax.set_xlabel("m/z"); ax.set_ylabel("Relative intensity (%)")
    entropy = float(row.get("iceberg_entropy_similarity", 0) or 0)
    cosine = float(row.get("iceberg_cosine_similarity", 0) or 0)
    ax.set_title(f"{candidate_id}: experimental vs ICEBERG predicted MS/MS\nEntropy {entropy:.3f} | Cosine {cosine:.3f}")
    ax.text(.01, .96, "Experimental ↑", transform=ax.transAxes, color="#d73027", va="top")
    ax.text(.01, .04, "ICEBERG predicted ↓", transform=ax.transAxes, color="#2166ac", va="bottom")
    fig.tight_layout()
    pdf = out_dir / f"{candidate_id}_experimental_vs_ICEBERG.pdf"
    png = out_dir / f"{candidate_id}_experimental_vs_ICEBERG.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=250, bbox_inches="tight"); plt.close(fig)
    return [pdf, png]


def _fragment_mol(parent, frag_id):
    atoms = [i for i in range(parent.GetNumAtoms()) if (int(frag_id) >> i) & 1]
    if not atoms:
        return None
    smiles = Chem.MolFragmentToSmiles(parent, atomsToUse=atoms, canonical=True)
    mol = Chem.MolFromSmiles(smiles)
    # A retained aromatic atom subset may be chemically meaningful to ICEBERG
    # but fail standalone aromatic sanitization.  It is still drawable and
    # should not make the matched-fragment report disappear.
    return mol if mol is not None else Chem.MolFromSmiles(smiles, sanitize=False)


def _fragment_key(mol):
    """Return a stable de-duplication key without requiring sanitization."""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        # Unsanitized aromatic subsets can be drawable even when RDKit cannot
        # convert them back to a canonical SMILES string.
        return Chem.MolToMolBlock(mol, kekulize=False)


def _draw_fragment_image(mol, size=(700, 430)):
    """Draw an ICEBERG fragment, tolerating non-kekulizable aromatic subsets."""
    try:
        return Draw.MolToImage(mol, size=size, kekulize=True)
    except Exception:
        return Draw.MolToImage(mol, size=size, kekulize=False)


def _fragments(candidate_id, row, spectrum, pred, out_dir):
    emz, _ = _peaks(spectrum)
    pmz = np.asarray(pred.get("mz", []), float); pint = np.asarray(pred.get("intensity", []), float)
    frag_ids = np.asarray(pred.get("frag_ids", []))
    parent = Chem.MolFromSmiles(str(pred.get("canonical_smiles") or row.get("smiles") or ""))
    if parent is None or not len(emz) or len(frag_ids) != len(pmz):
        return []
    cards = []
    for j, mass in enumerate(pmz):
        i = int(np.argmin(np.abs(emz - mass))); error = float(emz[i] - mass)
        if abs(error) > max(0.01, float(mass) * 10e-6):
            continue
        try:
            mol = _fragment_mol(parent, frag_ids[j])
        except Exception:
            # A malformed individual fragment must not suppress structures
            # for the other matched peaks in this candidate.
            continue
        if mol is not None:
            cards.append((float(pint[j]), float(mass), float(emz[i]), error, mol))
    cards.sort(reverse=True, key=lambda x: x[0])
    unique, seen = [], set()
    for card in cards:
        try:
            key = (round(card[1], 6), _fragment_key(card[4]))
        except Exception:
            key = (round(card[1], 6), int(card[4].GetNumAtoms()))
        if key not in seen:
            seen.add(key); unique.append(card)
    pdf = out_dir / f"{candidate_id}_all_experiment_matched_ICEBERG_fragments.pdf"
    png = out_dir / f"{candidate_id}_all_experiment_matched_ICEBERG_fragments.png"
    if not unique:
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        ax.text(.5, .58, candidate_id, ha="center", fontsize=22, weight="bold")
        ax.text(.5, .45,
                "No ICEBERG fragment structure matched an experimental peak\n"
                "within max(10 ppm, 0.01 Da).",
                ha="center", va="center", fontsize=14)
        fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return [pdf, png]
    with PdfPages(pdf) as pages:
        for start in range(0, len(unique), 12):
            subset = unique[start:start + 12]; fig, axes = plt.subplots(4, 3, figsize=(15, 19))
            for ax in axes.flat: ax.axis("off")
            for ax, (inten, theory, observed, error, mol) in zip(axes.flat, subset):
                try:
                    ax.imshow(_draw_fragment_image(mol, size=(700, 430)))
                except Exception as exc:
                    # Preserve the report and the peak metadata even if one
                    # pathological RDKit molecule cannot be rendered at all.
                    ax.text(.5, .5, f"Fragment structure unavailable\n{type(exc).__name__}",
                            transform=ax.transAxes, ha="center", va="center", fontsize=10)
                ax.axis("off")
                try:
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                except Exception:
                    formula = "unsanitized ICEBERG substructure"
                ax.set_title(f"ICEBERG m/z {theory:.5f} | predicted {100*inten/max(float(pint.max()),1e-12):.1f}%", fontsize=9)
                ax.text(.5, -.02, f"{formula}\nexperiment {observed:.5f} | Δ {error:+.5f} Da",
                        transform=ax.transAxes, ha="center", va="top", fontsize=8)
            fig.suptitle(f"{candidate_id}: all experimental-matched ICEBERG fragment structures", fontsize=17)
            fig.tight_layout(rect=(0, 0, 1, .97)); pages.savefig(fig, bbox_inches="tight")
            if start == 0: fig.savefig(png, dpi=220, bbox_inches="tight")
            plt.close(fig)
    return [pdf, png]


def export_result_figures(results, spectra, prediction_cache, output_root, full_results=None, progress=print):
    """Export mirror spectra and matched-fragment structure sheets."""
    root = Path(output_root); frag_dir = root / "figures" / "iceberg_fragments"
    mirror_dir = root / "figures" / "spectra_comparison"
    frag_dir.mkdir(parents=True, exist_ok=True); mirror_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ordinal, (_, row) in enumerate(results.reset_index(drop=True).iterrows(), 1):
        candidate_id = _candidate_id(row, ordinal)
        try:
            spectrum = spectra[int(row.get("spectrum_index", -1))]
            pred = _cache_for(row, ordinal, full_results, prediction_cache) or {}
            made += _mirror(candidate_id, row, spectrum, pred, mirror_dir)
            made += _fragments(candidate_id, row, spectrum, pred, frag_dir)
        except Exception as exc:
            progress(f"[figures] {candidate_id} skipped: {exc}")
    progress(f"[figures] generated {len(made)} files under {root / 'figures'}")
    return made
