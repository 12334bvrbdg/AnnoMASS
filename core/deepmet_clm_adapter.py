"""Adapter for AnnoMASS's parallel DeepMet CLM candidate route.

DeepMet's flavonoid CLM is an unconditional molecular generator: the MS/MS
spectrum is not an input token to the RNN.  The deployable artifact is therefore
the generated, strictly validated candidate pool.  This adapter mass-matches
every input spectrum against that pool, independently of whether its formula is
already represented in AnnoMASS's structure database.

Only structures absent from the active AnnoMASS database are returned.  These
rows are subsequently scored by the same ICEBERG/Spectrum Predictor pass as
database candidates.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi


from core.paths import external_dir as _external_dir

DEEPMET_ROOT = _external_dir("deepmet")
DEFAULT_CANDIDATE_POOL = DEEPMET_ROOT / "final_results/flavonoid_candidates_final.csv"
REQUIRED_COLUMNS = {
    "canonical_smiles", "molecular_formula", "exact_mass", "sampling_frequency",
}


def is_available(candidate_pool: str | Path = DEFAULT_CANDIDATE_POOL) -> bool:
    return Path(candidate_pool).is_file()


def _canonical_smiles(value: object) -> str:
    try:
        mol = Chem.MolFromSmiles(str(value or "").strip())
        if mol is None:
            return ""
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return ""


@lru_cache(maxsize=4)
def _load_pool_cached(path_text: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns  # cache invalidation is encoded in the function key
    frame = pd.read_csv(path_text, low_memory=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"DeepMet CLM candidate pool missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["canonical_smiles"] = frame["canonical_smiles"].map(_canonical_smiles)
    frame["exact_mass"] = pd.to_numeric(frame["exact_mass"], errors="coerce")
    frame["sampling_frequency"] = pd.to_numeric(
        frame["sampling_frequency"], errors="coerce"
    ).fillna(0)
    frame = frame[
        frame["canonical_smiles"].ne("") & frame["exact_mass"].notna()
    ].drop_duplicates("canonical_smiles")
    return frame.sort_values("exact_mass", kind="stable").reset_index(drop=True)


def load_candidate_pool(candidate_pool: str | Path = DEFAULT_CANDIDATE_POOL) -> pd.DataFrame:
    path = Path(candidate_pool).resolve()
    return _load_pool_cached(str(path), path.stat().st_mtime_ns)


def database_structure_keys(database: pd.DataFrame) -> set[str]:
    if database is None or database.empty:
        return set()
    for column in ("CanonicalSMILES", "canonical_smiles", "SMILES", "smiles"):
        if column in database.columns:
            return {s for s in database[column].map(_canonical_smiles) if s}
    return set()


def inchikey_from_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return inchi.MolToInchiKey(mol) if mol is not None else ""
    except Exception:
        return ""


def match_neutral_mass(
    neutral_mass: float,
    database_keys: set[str],
    ppm_tolerance: float = 10.0,
    top_n: int = 10,
    candidate_pool: str | Path = DEFAULT_CANDIDATE_POOL,
) -> pd.DataFrame:
    """Return CLM-generated structures in the precursor mass window.

    Formula equality is deliberately not part of this API.  The only spectrum
    prefilter is the measured neutral-mass window needed to keep ICEBERG work
    tractable.
    """
    try:
        mass = float(neutral_mass)
    except (TypeError, ValueError):
        return pd.DataFrame()
    if not np.isfinite(mass) or mass <= 0:
        return pd.DataFrame()

    pool = load_candidate_pool(candidate_pool)
    masses = pool["exact_mass"].to_numpy(float)
    delta = mass * float(ppm_tolerance) / 1e6
    left = int(np.searchsorted(masses, mass - delta, side="left"))
    right = int(np.searchsorted(masses, mass + delta, side="right"))
    result = pool.iloc[left:right].copy()
    if database_keys:
        result = result[~result["canonical_smiles"].isin(database_keys)]
    if result.empty:
        return result
    result["mass_error_ppm"] = (result["exact_mass"] - mass) / mass * 1e6
    result = result.sort_values(
        ["sampling_frequency", "mass_error_ppm"],
        ascending=[False, True], kind="stable",
        key=lambda col: col.abs() if col.name == "mass_error_ppm" else col,
    )
    if int(top_n) > 0:
        result = result.head(int(top_n))
    return result.reset_index(drop=True)
