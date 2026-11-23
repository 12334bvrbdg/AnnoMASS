"""DreaMS adapter for GUI integration.

Runs DreaMS search via a subprocess in the 'dreams' conda environment
(which has all required heavy dependencies: torch, lightning, sklearn, etc.)
using the same workflow as the 谱图对比.ipynb notebook:
  MSData.load → dreams_embeddings → cosine_similarity → top-k → Modified Cosine

Exports:
    list_reference_libraries() -> list[str]
    spectra_to_mgf(spectrums, out_path) -> int
    run_dreams_search(...) -> pd.DataFrame
    DEFAULT_REF_HDF5 : Path
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
from core.paths import external_dir as _external_dir, env_python as _env_python, ENV_RETRIEVAL

_DREAMS_ROOT = _external_dir("DreaMS")
REFERENCE_LIBRARY_DIR = _DREAMS_ROOT / "data" / "reference_libraries"
_LOCAL_REFERENCE_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "data" / "reference_libraries"
# Default is the flavonoid-scoped variant (MassSpecGym_DreaMS_flavonoid_only.hdf5,
# 24,934 spectra, SMARTS-selected; contains the same 332-spectrum/77-compound
# deepmass_flavonoid supplement as the full library -- verified identical sets
# 2026-09-16). Switched 2026-09-16 from the 231,436-spectrum augmented general
# library (MassSpecGym train/val/test + the same 332 supplement): the two were
# previously verified result-identical on the 522 benchmark (full-pipeline A/B,
# pre-topk-fix: Top-1 both 65.7%, zero per-query differences) since every
# DreaMS-route candidate is independently flavonoid-class-gated regardless of
# library composition -- so this switch is for search-space/latency (10.8% of
# the entries) and to remove the theoretical risk of a true flavonoid neighbor
# being pushed past DreaMS's own internal top-N cutoff by unrelated generic
# entries before ever reaching that gate. Re-verify against GNPS-522/MoNA-689
# before trusting new numbers if this default is ever reverted.
_MANUAL_REF_HDF5 = _LOCAL_REFERENCE_LIBRARY_DIR / "MassSpecGym_DreaMS_flavonoid_only.hdf5"
DEFAULT_REF_HDF5 = (
    _MANUAL_REF_HDF5
    if _MANUAL_REF_HDF5.exists()
    else REFERENCE_LIBRARY_DIR / "MassSpecGym_DreaMS.hdf5"
)
_DREAMS_PYTHON = os.environ.get("ANNOMASS_RETRIEVAL_PYTHON") or _env_python(ENV_RETRIEVAL, "dreams")
_WORKER_SCRIPT = str(Path(__file__).parent / "_dreams_search_worker.py")


def _format_mgf_charge(charge) -> str:
    try:
        z = int(charge)
    except Exception:
        text = str(charge or "").strip()
        return text if text else "1+"
    if z < 0:
        return f"{abs(z)}-"
    return f"{abs(z) or 1}+"


def list_reference_libraries() -> list[str]:
    libs = []
    if REFERENCE_LIBRARY_DIR.exists():
        libs.extend(REFERENCE_LIBRARY_DIR.glob("*.hdf5"))
    if _LOCAL_REFERENCE_LIBRARY_DIR.exists():
        libs.extend(_LOCAL_REFERENCE_LIBRARY_DIR.glob("*.hdf5"))
    return sorted(str(p) for p in libs)


def spectra_to_mgf(spectrums, out_path: str, return_mapping: bool = False):
    """Export Spectrum objects to MGF.

    Returns number of spectra written by default. When return_mapping=True,
    returns (written, source_indices, titles), where source_indices maps each
    DreaMS query_index back to the input spectrums list index.
    """
    written = 0
    source_indices = []
    titles = []
    with open(out_path, "w", encoding="utf-8") as f:
        for src_i, sp in enumerate(spectrums):
            # matchms Spectrum stores peaks in sp.peaks.mz / sp.peaks.intensities
            try:
                mz = sp.peaks.mz
                intensity = sp.peaks.intensities
            except AttributeError:
                mz = sp.get("mz")
                intensity = sp.get("intensity")
            if mz is None or intensity is None or len(mz) == 0:
                continue
            precursor_mz = sp.get("precursor_mz", 0.0) or 0.0
            title = sp.get("spectrumid") or sp.get("title") or f"spec_{written}"
            f.write("BEGIN IONS\n")
            f.write(f"TITLE={title}\n")
            f.write(f"PEPMASS={float(precursor_mz):.6f}\n")
            charge = sp.get("charge", 1)
            if charge:
                f.write(f"CHARGE={_format_mgf_charge(charge)}\n")
            adduct = sp.get("adduct") or sp.get("ion") or sp.get("ADDUCT") or sp.get("ION") or ""
            if adduct:
                f.write(f"ION={str(adduct).strip()}\n")
                f.write(f"ADDUCT={str(adduct).strip()}\n")
            ionmode = sp.get("ionmode", "")
            if ionmode:
                f.write(f"IONMODE={ionmode}\n")
            rt = sp.get("retention_time") or sp.get("rt") or 0.0
            f.write(f"RTINSECONDS={float(rt):.3f}\n")
            for m, i in zip(mz, intensity):
                f.write(f"{m:.4f} {i:.4f}\n")
            f.write("END IONS\n\n")
            source_indices.append(src_i)
            titles.append(str(title))
            written += 1
    if return_mapping:
        return written, source_indices, titles
    return written


def run_dreams_search(
    mgf_path: str,
    ref_hdf5: str = str(DEFAULT_REF_HDF5),
    top_n: int = 10,
    cho_only: bool = False,
    mz_tol_ppm: float = 0.0,
    batch_size: int = 32,
    progress_callback=None,
    mz_tol_ppm_per_query=None,
) -> pd.DataFrame:
    """Run DreaMS notebook-style search via the 'dreams' conda subprocess.

    Returns a DataFrame with columns from the notebook result including
    DreaMS_similarity and Modified_cosine_similarity.
    mz_tol_ppm_per_query: optional list (MGF order) overriding mz_tol_ppm per query.
    Raises on error.
    """
    mgf_path = str(mgf_path)
    ref_hdf5 = str(ref_hdf5)
    out_csv  = mgf_path + "_dreams_results.csv"

    if not os.path.isfile(mgf_path):
        raise FileNotFoundError(f"MGF not found: {mgf_path}")
    if not os.path.isfile(ref_hdf5):
        raise FileNotFoundError(f"Reference HDF5 not found: {ref_hdf5}")
    if not os.path.isfile(_DREAMS_PYTHON):
        raise FileNotFoundError(
            f"DreaMS Python not found: {_DREAMS_PYTHON}\n"
            "Activate or install the 'dreams' conda environment."
        )

    cfg = {
        "mgf_path":    mgf_path,
        "ref_hdf5":    ref_hdf5,
        "top_n":       top_n,
        "cho_only":    cho_only,
        "mz_tol_ppm":  float(mz_tol_ppm or 0.0),
        "out_csv":     out_csv,
        "batch_size":  batch_size,
        "dreams_root": str(_DREAMS_ROOT),
    }
    if mz_tol_ppm_per_query is not None:
        tol_path = mgf_path + "_mz_tol_ppm.json"
        with open(tol_path, "w") as fh:
            json.dump([float(x) for x in mz_tol_ppm_per_query], fh)
        cfg["mz_tol_ppm_file"] = tol_path

    if progress_callback:
        progress_callback("Launching DreaMS subprocess...")

    try:
        os.remove(out_csv)
    except FileNotFoundError:
        pass

    recent_output = deque(maxlen=80)
    proc = subprocess.Popen(
        [_DREAMS_PYTHON, _WORKER_SCRIPT, json.dumps(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        clean_line = line.replace("[DreaMS] ", "")
        recent_output.append(clean_line)
        if progress_callback:
            progress_callback(clean_line)
    proc.wait()

    if proc.returncode != 0:
        tail = "\n".join(recent_output)
        details = f"\nLast DreaMS output:\n{tail}" if tail else ""
        raise RuntimeError(
            f"DreaMS subprocess exited with code {proc.returncode}.{details}"
        )

    if not os.path.isfile(out_csv):
        return pd.DataFrame()

    df = pd.read_csv(out_csv)
    if progress_callback:
        progress_callback(f"Done: {len(df)} hits for {df['query_index'].nunique()} spectra")
    return df
