"""AnnoMASS staged identification algorithm (positive ion).

Single implementation of the AnnoMASS pipeline, called by the GUI worker
thread, ``run_pipeline_headless.py`` and ``public_pip_ablation.py``.

Every spectrum walks a fixed waterfall and stops at the first level that
yields a passing candidate (see core/staged_levels.py for the scoring rules):

  A  DreaMS library retrieval     DreaMS similarity (+) direct spectral cosine
  B  DeepMASS candidates + DreaMS hits that failed A, with library evidence
                                  DeepMASS score (+) direct spectral cosine (+) DreaMS score
  C  every candidate (also those that failed B), carried B score
                                  (+) ICEBERG predicted-spectrum similarity
  D  DeepMet CLM structure absent from the structure DB
                                  ICEBERG entropy + cosine dual Top-1
  -  otherwise the spectrum is not annotated.

Then: AnnoSM gated rerank inside each spectrum -> compound dedup -> Top-K ->
optional ABCoRT retention time.  Negative-ion spectra are skipped.
"""

import math
import os
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from molmass import Formula

from core.main import identify_unknown
from core.flavclass_filter import classify as flavonoid_classify
from core.icefrag_adapter import (
    is_available as mf_available,
    predict_batch as mf_predict_batch,
)
from core import staged_levels as _levels
from core.staged_levels import select_topk_per_spectrum as _select_topk_per_spectrum
from core.flavonoid_fragment_scorer import (
    load_loss_table as _load_fragment_loss_table,
    score_candidate as _score_fragment_evidence,
)


# Pipeline is a flavonoid workflow -- keep the class set narrow so generic
# chromone hits never pass as flavonoids.
STRICT_FLAV_CLASSES = {
    "Flavone", "Flavonol", "Flavanone", "Flavanonol", "Dihydroflavonol",
    "Isoflavone", "Isoflavanone", "Isoflavan", "Flavan", "Flavan3ol",
    "Chalcone", "Dihydrochalcone", "Aurone", "Anthocyanidin", "Pterocarpan",
    "IsoflavanChroman",
}

# De novo-only sanity gate, kept OUT of core/flavclass_filter.py so the shared
# classifier is untouched. A generative CLM can emit chemically implausible
# structures that still match a flavonoid subclass SMARTS.
_LOOSE_CHALCONE_SMARTS = "O=C(/C=C/c1ccccc1)c1ccccc1"
_OPEN_CHALCONE_SMARTS = "O=[C;R0](/C=C/c1ccccc1)c1ccccc1"

# Keep CPU headroom for the Qt UI / external model subprocesses.
MAX_WORKERS = min(max(os.cpu_count() or 1, 2), 4)
# CLM route and fragment-poor gate: a spectrum needs this many real fragments.
_MIN_EFFECTIVE_FRAGMENTS = 3


def _candidate_structure_dedup_identity(smiles="", inchikey="", name="", formula="", row_id=""):
    """Return a stereochemistry-aware identity for final candidate dedup.

    Structure is deliberately preferred over display name: library records for
    different stereoisomers commonly share a compound name, while an isomeric
    SMILES can distinguish them. A full InChIKey generated from the SMILES is
    the most stable key across alternate SMILES spellings.
    """
    def _clean(value):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        value = str(value or "").strip()
        return "" if value.lower() in ("nan", "none", "null") else value

    smi = _clean(smiles)
    if smi:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                full_ikey = Chem.MolToInchiKey(mol).upper()
                if full_ikey:
                    return "ik:" + full_ikey
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                if canonical:
                    return "smi:" + canonical
        except Exception:
            pass

    supplied_ikey = _clean(inchikey).upper()
    if re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", supplied_ikey):
        return "ik:" + supplied_ikey
    if smi:
        return "smi:" + smi

    clean_name = _clean(name)
    if clean_name:
        return "name:" + re.sub(r"\s+", " ", clean_name).casefold()
    clean_formula = _clean(formula)
    if clean_formula:
        return "formula:" + clean_formula
    return "row:" + str(row_id)


def _candidate_route_dedup_identity(row, per_spectrum=False):
    """Return a dedup key without merging CLM into the library route."""
    spectrum_prefix = (
        f"sp{row.get('spectrum_index', '')}|" if per_spectrum else ""
    )
    route_prefix = (
        "clm|"
        if str(row.get("candidate_source", "") or "") == "deepmet_clm"
        else "library|"
    )
    structure_identity = _candidate_structure_dedup_identity(
        smiles=row.get("smiles", ""),
        inchikey=row.get("inchikey", ""),
        name=row.get("compound_name", ""),
        formula=row.get("formula", ""),
        row_id=getattr(row, "name", ""),
    )
    return spectrum_prefix + route_prefix + structure_identity


def _formula_key(value):
    """Normalize a molecular formula for library-presence comparisons."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    return re.sub(r"[+-]+$", "", text)


def _structure_library_formula_counts(database):
    """Return formula -> compound-row count for the active structure library."""
    if database is None or database.empty:
        return {}
    formula_col = next(
        (c for c in ("Formula", "formula", "MolecularFormula", "molecular_formula")
         if c in database.columns),
        None,
    )
    if formula_col is None:
        return {}
    keys = database[formula_col].map(_formula_key)
    return keys[keys.ne("")].value_counts().astype(int).to_dict()


def _is_plausible_denovo_structure(smiles):
    """Reject de novo structures that are chemically implausible even though
    they matched a flavonoid subclass SMARTS: triple bonds (alkynes/nitriles),
    cumulated double bonds (allenes/cumulenes), and enones that only match the
    chalcone pattern with a ring-fused carbonyl (a true chalcone is open-chain).
    """
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.TRIPLE:
            return False
    for atom in mol.GetAtoms():
        n_double = sum(1 for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        if n_double >= 2:
            return False
    loose = Chem.MolFromSmarts(_LOOSE_CHALCONE_SMARTS)
    open_chalcone = Chem.MolFromSmarts(_OPEN_CHALCONE_SMARTS)
    if loose is not None and open_chalcone is not None:
        if mol.HasSubstructMatch(loose) and not mol.HasSubstructMatch(open_chalcone):
            return False
    return True


# ── DreaMS lazy adapter (Qt-free) ────────────────────────────────────────
_DREAMS_AVAILABLE = None
_DREAMS_IMPORT_ERROR = None
dreams_spectra_to_mgf = None
run_dreams_search = None


def _ensure_dreams_adapter():
    global _DREAMS_AVAILABLE, _DREAMS_IMPORT_ERROR, dreams_spectra_to_mgf, run_dreams_search
    if _DREAMS_AVAILABLE is True:
        return True
    if _DREAMS_AVAILABLE is False:
        return False
    try:
        from core.dreams_adapter import (
            spectra_to_mgf as _spectra_to_mgf,
            run_dreams_search as _run_dreams_search,
        )
        dreams_spectra_to_mgf = _spectra_to_mgf
        run_dreams_search = _run_dreams_search
        _DREAMS_AVAILABLE = True
        return True
    except Exception as _e:
        _DREAMS_AVAILABLE = False
        _DREAMS_IMPORT_ERROR = str(_e)
        print(f"[DreaMS] Not available: {_e}")
        return False


# ── DeepMASS experimental spectral library (level B) ─────────────────────
class LibrarySpectrumIndex:
    """InChIKey14 -> experimental reference spectra of the DeepMASS positive
    spectral library, for the level-B direct spectral match.

    *exclude* holds library positions to ignore (benchmark near-duplicate
    blinding; empty in normal use)."""

    def __init__(self, references, exclude=(), per_compound=40):
        self.references = references
        excluded = set(int(x) for x in exclude)
        grouped = defaultdict(list)
        for pos, ref in enumerate(references or []):
            if pos in excluded:
                continue
            try:
                ik = str(ref.get("inchikey") or "")[:14].upper()
            except Exception:
                continue
            if len(ik) == 14:
                grouped[ik].append(pos)
        if per_compound and per_compound > 0:
            for ik, positions in grouped.items():
                if len(positions) > per_compound:
                    positions.sort(key=lambda p: -len(references[p].peaks.mz))
                    grouped[ik] = positions[:per_compound]
        self.by_key = dict(grouped)
        from matchms.similarity import ModifiedCosine
        self._cosine = ModifiedCosine(tolerance=0.05)

    def has(self, inchikey) -> bool:
        return str(inchikey or "")[:14].upper() in self.by_key

    def best_cosine(self, query, inchikey):
        """(best modified cosine, number of reference spectra) for the
        compound; (0.0, 0) when it has no library spectrum."""
        positions = self.by_key.get(str(inchikey or "")[:14].upper(), [])
        best = 0.0
        for pos in positions:
            try:
                score = self._cosine.pair(query, self.references[pos])
                try:
                    value = float(score["score"])
                except (IndexError, KeyError, TypeError, ValueError):
                    value = float(score)
            except Exception:
                continue
            if value == value and value > best:
                best = value
        return best, len(positions)

    def best_match(self, query, inchikey):
        """(best modified cosine, number of reference spectra, peaks string
        "mz:rel_intensity;..." of the best-matching reference)."""
        positions = self.by_key.get(str(inchikey or "")[:14].upper(), [])
        best, best_pos = 0.0, None
        for pos in positions:
            try:
                score = self._cosine.pair(query, self.references[pos])
                try:
                    value = float(score["score"])
                except (IndexError, KeyError, TypeError, ValueError):
                    value = float(score)
            except Exception:
                continue
            if value == value and value > best:
                best, best_pos = value, pos
        peaks = ""
        if best_pos is not None:
            try:
                ref = self.references[best_pos]
                mz = np.asarray(ref.peaks.mz, dtype=float)
                it = np.asarray(ref.peaks.intensities, dtype=float)
                if len(it) and it.max() > 0:
                    keep = np.argsort(it)[-60:]
                    keep = keep[np.argsort(mz[keep])]
                    peaks = ";".join(f"{mz[k]:.4f}:{it[k] / it.max():.4f}" for k in keep)
            except Exception:
                peaks = ""
        return best, len(positions), peaks


@dataclass
class AnnoMassParams:
    """Every tunable knob of the AnnoMASS staged algorithm.

    The level weights/thresholds were chosen on the benchmark-disjoint
    calibration set (scripts/calibration/calibrate_stage_weights.py;
    reports/calibration/).  A weight is the share of the first score in each
    level's formula (see core/staged_levels.py)."""
    mz_tol_ppm: float = 10.0
    cho_only: bool = True
    # v5 (2026-10-30).  Level C/B thresholds, the C weight and the score modes come from the
    # benchmark-disjoint calibration set (reports/calibration/, 334 positive spectra, blinded libraries).
    # The DreaMS share of A and B is a design choice (DreaMS is the most accurate single source and its
    # embedding is the most robust across instruments): the calibration set alone preferred wA = 0.15,
    # but that loses DreaMS's correct answers whenever cross-instrument library cosines are low.
    # wA = 0.8 / tA = 0.70 was taken from the region of an offline grid in which no main-benchmark Top-1
    # dropped (tA 0.60-0.73 all qualify, centre 0.70; reports/staged_dev/weight_grid_offline.csv).  B keeps
    # DeepMASS as its largest share (0.5 in v7, user choice; 0.6 in v6) with DreaMS carried at 0.3.
    # level A: pooled DreaMS similarity (weight) + pooled modified cosine (1 - weight)
    level_a_weight: float = 0.80
    level_a_threshold: float = 0.70
    # level B: DeepMASS score (level_b_weight) + library cosine (level_b_library_weight)
    #          + DreaMS similarity (the remainder 1 - both = 0.3); v7: 0.5 / 0.2 / 0.3 (v6: 0.6 / 0.1 / 0.3)
    level_b_weight: float = 0.50
    level_b_library_weight: float = 0.20
    level_b_threshold: float = 0.97
    # level C: carried B score (weight) + ICEBERG similarity (1 - weight)
    level_c_weight: float = 0.50
    level_c_threshold: float = 0.12
    deepmass_score_mode: str = "consensus"   # "consensus" | "absolute"
    level_c_ice_metric: str = "ice_cosine"   # "mf_cosine" | "ice_cosine" | "ice_entropy"
    # level D: DeepMet CLM dual Top-1
    # D passes when ONE candidate tops both ICEBERG metrics and (entropy >= primary and cosine >= secondary) OR (cosine >= primary and entropy >= secondary)
    clm_primary_min: float = 0.5
    clm_secondary_min: float = 0.4
    # DeepMASS candidates per spectrum entering B/C (0 = every candidate)
    deepmass_candidate_top_n: int = 0
    library_spectra_per_compound: int = 40
    # positions of the DeepMASS spectral library to ignore (benchmark blinding only)
    ref_pos_exclude: Optional[str] = None
    # calibration: compute every stage for every spectrum and dump the table
    evaluate_all_stages: bool = False
    # modules to run, any subset of "ABCD" in waterfall order (A DreaMS, B DeepMASS + library spectra,
    # C DeepMASS + ICEBERG, D DeepMet CLM); skipped modules are not computed.  D additionally needs
    # use_deepmet_clm.
    enabled_levels: str = "ABCD"
    # Spectrum Predictor / ICEBERG
    mf_adduct: str = "[M+H]+"
    mf_nce: float = 30.0
    mf_instrument: str = "Unknown"
    iceberg_peak_min_rel_intensity: float = 0.01
    iceberg_peak_top_k: int = 80
    # output
    top_n_per_spectrum: int = 0
    dedup_per_spectrum: bool = False
    benchmark_all_ms2: bool = False
    # A compound is only reported if at least one of its supporting spectra has
    # this many real fragment peaks (>=1% base, >2 Da from precursor). 0 disables.
    min_fragment_peaks: int = 3
    # DreaMS search
    dreams_top_n: int = 20
    dreams_batch_size: int = 32
    # AnnoSM rerank (inside each spectrum's level)
    enable_annosm_rerank: bool = True
    # AnnoSM may only promote a challenger whose level score is within this gap of the rank-1
    # (2026-10-30: unrestricted promotions were mostly wrong under the DreaMS-weighted A level)
    annosm_max_level_gap: float = 0.02
    annosm_include_external: bool = True
    # ABCoRT retention time
    enable_abcort_predict: bool = False
    abcort_checkpoint_path: Optional[str] = None
    # DeepMet CLM route
    use_deepmet_clm: bool = False
    deepmet_clm_top_n: int = 0
    deepmet_clm_mass_tol_ppm: float = 10.0
    evidence_score_flag_threshold: float = 0.3

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.type in (float, "float"):
                setattr(self, f.name, float(value))
            elif f.type in (int, "int"):
                setattr(self, f.name, int(value))
            elif f.type in (bool, "bool"):
                setattr(self, f.name, bool(value))
        for name in ("level_a_weight", "level_b_weight", "level_b_library_weight",
                     "level_c_weight", "iceberg_peak_min_rel_intensity"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.level_b_weight + self.level_b_library_weight > 1.0 + 1e-9:
            raise ValueError("level_b_weight + level_b_library_weight must not exceed 1")
        if self.deepmass_score_mode not in ("consensus", "absolute"):
            raise ValueError("deepmass_score_mode must be 'consensus' or 'absolute'")
        if self.level_c_ice_metric not in ("mf_cosine", "ice_cosine", "ice_entropy"):
            raise ValueError("level_c_ice_metric must be mf_cosine, ice_cosine or ice_entropy")
        if self.iceberg_peak_top_k < 0:
            raise ValueError("iceberg_peak_top_k must be non-negative")

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Build params from keyword arguments, ignoring options of the
        retired orthogonal-cascade architecture (dreams_sim_threshold,
        pterocarpan_*, deepmass_only_*, library_iceberg_*, mf_top_n, ...)."""
        known = {f.name for f in fields(cls)}
        retired = sorted(k for k in kwargs if k not in known)
        if retired:
            print(f"[AnnoMassParams] ignoring retired options: {', '.join(retired)}")
        return cls(**{k: v for k, v in kwargs.items() if k in known})


def _is_negative_spectrum(sp):
    ionmode = str(sp.get("ionmode") or sp.get("IONMODE") or "").lower()
    adduct = str(
        sp.get("adduct") or sp.get("ion") or sp.get("ADDUCT") or sp.get("ION") or ""
    ).strip()
    return ionmode.startswith("neg") or adduct.endswith("-")


def _ms2_peak_count(sp):
    try:
        return len(sp.peaks.mz)
    except Exception:
        try:
            return len(sp.get("mz") or [])
        except Exception:
            return 0


def _effective_fragment_count(sp, min_rel=0.01, precursor_exclusion_da=2.0):
    """MS2 peaks that are real fragments: >= min_rel of the base peak and more
    than precursor_exclusion_da away from the precursor."""
    try:
        mz = np.asarray(sp.peaks.mz, float)
        ints = np.asarray(sp.peaks.intensities, float)
    except Exception:
        try:
            mz = np.asarray(sp.get("mz") or [], float)
            ints = np.asarray(sp.get("intensity") or [], float)
        except Exception:
            return 0
    if len(mz) == 0 or len(ints) == 0:
        return 0
    base = float(np.nanmax(ints))
    if base <= 0:
        return 0
    keep = np.isfinite(mz) & np.isfinite(ints) & (ints >= min_rel * base)
    prec = float(sp.get("precursor_mz") or 0.0)
    if prec > 0:
        keep &= (np.abs(mz - prec) > precursor_exclusion_da)
    return int(np.sum(keep))


def _spectrum_adduct(sp):
    adduct = (
        sp.get("adduct") or sp.get("precursor_type") or sp.get("precursortype") or
        sp.get("ion") or sp.get("ADDUCT") or sp.get("PRECURSOR_TYPE") or
        sp.get("PRECURSORTYPE") or sp.get("ION") or ""
    )
    adduct = str(adduct or "").strip()
    if adduct.lower() in ("nan", "none", "null"):
        adduct = ""
    return adduct or "[M+H]+"


def _spectrum_title(sp, i):
    return sp.get("spectrumid") or sp.get("title") or f"spec_{i}"


_SMI_CACHE = {}
_FORMULA_MASS_CACHE = {}


def _norm_smi(v):
    smi = str(v or "").strip()
    if not smi or smi.lower() in ("nan", "none", "null"):
        return ""
    if smi not in _SMI_CACHE:
        try:
            mol = Chem.MolFromSmiles(smi)
            _SMI_CACHE[smi] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else smi
        except Exception:
            _SMI_CACHE[smi] = smi
    return _SMI_CACHE[smi]


def _norm_ikey(v):
    s = str(v or "").strip().upper()
    return "" if not s or s in ("NAN", "NONE", "NULL") else s


def _norm_formula(v):
    f = str(v or "").strip()
    if not f or f.lower() in ("nan", "none", "null"):
        return ""
    return re.sub(r"[+-]+$", "", f)


def _formula_mass(formula):
    f = _norm_formula(formula)
    if not f:
        return None
    if f not in _FORMULA_MASS_CACHE:
        try:
            _FORMULA_MASS_CACHE[f] = float(Formula(f).isotope.mass)
        except Exception:
            _FORMULA_MASS_CACHE[f] = None
    return _FORMULA_MASS_CACHE[f]


def precursor_tolerance_ppm(precursor_mz, base_ppm):
    """MS1 tolerance for one spectrum: *base_ppm* plus the rounding error of the
    precursor m/z as recorded.  Libraries often store the precursor with only
    2-3 decimals (285.07 for 285.0757 = 20 ppm), which a fixed 10 ppm window
    would reject; with 4+ decimals the extra term is below 0.2 ppm."""
    try:
        mz = float(precursor_mz)
    except (TypeError, ValueError):
        return base_ppm
    if not mz > 0:
        return base_ppm
    text = repr(round(mz, 6))
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return base_ppm + 0.5 * 10.0 ** (-decimals) / mz * 1e6


def _ppm_error(candidate_mass, neutral_mass):
    try:
        a, b = float(candidate_mass), float(neutral_mass)
        if a <= 0 or b <= 0:
            return None
        return (a - b) / b * 1e6
    except Exception:
        return None


def _strict_classes(smiles):
    return (set(flavonoid_classify(smiles)) & STRICT_FLAV_CLASSES) if smiles else set()


def _clean_text(v):
    s = str(v or "").strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def _row_identity(row):
    return _candidate_structure_dedup_identity(
        smiles=row.get("smiles", ""), inchikey=row.get("inchikey", ""),
        name=row.get("compound_name", ""), formula=row.get("formula", ""),
        row_id=getattr(row, "name", ""),
    )


def _score_with_iceberg(frame, identified, params, progress_cb, check_stop, label):
    """ICEBERG/Spectrum Predictor on every row of *frame* (in place): writes
    mf_score (cosine, 0.5 Da, precursor +-2 Da removed),
    iceberg_entropy_similarity and iceberg_cosine_similarity.  Returns
    {_pipeline_uid: predicted spectrum} for the GUI viewers."""
    cache = {}
    for col in ("mf_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity"):
        if col not in frame.columns:
            frame[col] = np.nan
    if frame.empty:
        return cache
    if not mf_available():
        progress_cb(f"{label}: Spectrum Predictor unavailable; predicted-spectrum scores stay 0.")
        frame[["mf_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity"]] = 0.0
        return cache
    from core.icefrag_adapter import _cosine_similarity, deepmet_entropy_cosine

    empty_pred = {"mz": np.array([]), "intensity": np.array([]),
                  "frag_ids": np.array([], dtype=np.int64), "frag_bits": 0,
                  "canonical_smiles": ""}
    groups = defaultdict(list)   # (adduct, nce) -> [(row index, spectrum index, smiles)]
    for idx, row in frame.iterrows():
        smiles = str(row.get("smiles", "") or "")
        if not smiles:
            continue
        sp_i = int(row["spectrum_index"])
        adduct = str(row.get("mf_adduct", "") or "").strip() or params.mf_adduct
        # Honour a per-spectrum collision energy when present (energy-sweep
        # benchmarks); otherwise the global params.mf_nce.
        nce = params.mf_nce
        try:
            raw = identified[sp_i].get("NCE") or identified[sp_i].get("nce") or identified[sp_i].get("ENERGY")
            if raw is not None and str(raw).strip() != "":
                nce = float(raw)
        except Exception:
            pass
        groups[(adduct, float(nce))].append((idx, sp_i, smiles))

    try:
        batch_size = max(1, int(os.environ.get("DEEPMASS_MF_BATCH_SMILES", "1000")))
    except Exception:
        batch_size = 1000
    obs_cache = {}
    for g_i, ((adduct, nce), entries) in enumerate(groups.items(), start=1):
        check_stop()
        unique_smiles = list(dict.fromkeys(smi for _, _, smi in entries))
        preds = []
        for start in range(0, len(unique_smiles), batch_size):
            check_stop()
            chunk = unique_smiles[start:start + batch_size]
            progress_cb(f"{label}: Spectrum Predictor {adduct} NCE {nce:g}, group {g_i}/{len(groups)}, "
                        f"{start + len(chunk)}/{len(unique_smiles)} structures")
            preds.extend(mf_predict_batch(smiles_list=chunk, adduct=adduct, nce=nce,
                                          instrument=params.mf_instrument, timeout=600))
        pred_by_smiles = {smi: (preds[k] if k < len(preds) else dict(empty_pred))
                          for k, smi in enumerate(unique_smiles)}
        for idx, sp_i, smi in entries:
            if sp_i not in obs_cache:
                sp = identified[sp_i]
                try:
                    mz = np.asarray(sp.peaks.mz, float)
                    ints = np.asarray(sp.peaks.intensities, float)
                except Exception:
                    mz = np.asarray(sp.get("mz") or [], float)
                    ints = np.asarray(sp.get("intensity") or [], float)
                obs_cache[sp_i] = (mz, ints, float(sp.get("precursor_mz") or 0.0))
            mz, ints, prec = obs_cache[sp_i]
            pred = pred_by_smiles.get(smi) or dict(empty_pred)
            pred_mz = np.asarray(pred["mz"], float)
            pred_int = np.asarray(pred["intensity"], float)
            # Precursor-peak exclusion (+-2 Da) for the scoring cosine only;
            # the cached spectrum for display keeps the full prediction.
            s_mz, s_int = pred_mz, pred_int
            if len(s_mz) > 0 and prec > 0:
                keep = np.abs(s_mz - prec) > 2.0
                s_mz, s_int = s_mz[keep], s_int[keep]
            frame.at[idx, "mf_score"] = float(_cosine_similarity(
                mz, ints, s_mz, s_int, tol_da=0.5, precursor_mz=prec))
            entropy, cosine = deepmet_entropy_cosine(
                mz, ints, pred_mz, pred_int, prec,
                min_rel_intensity=params.iceberg_peak_min_rel_intensity,
                top_k=params.iceberg_peak_top_k,
            )
            frame.at[idx, "iceberg_entropy_similarity"] = entropy
            frame.at[idx, "iceberg_cosine_similarity"] = cosine
            cache[int(frame.at[idx, "_pipeline_uid"])] = {
                "mz": pred_mz, "intensity": pred_int,
                "frag_ids": np.asarray(pred.get("frag_ids", np.array([], dtype=np.int64)), np.int64),
                "frag_bits": int(pred.get("frag_bits", 0)),
                "canonical_smiles": pred.get("canonical_smiles", ""),
            }
    for col in ("mf_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return cache


def _annotated_spectra(frame, params):
    """Spectrum indices that already have a passing candidate in *frame*."""
    if frame is None or frame.empty:
        return set()
    probe = frame.copy()
    passed = _levels.stage_pass(probe, params)
    return set(probe.loc[passed, "spectrum_index"].astype(int))


def run_annomass_pipeline(
    spectrums, pp, pn, deepmass_pos, deepmass_neg,
    ref_positive, ref_negative, default_database, ref_hdf5,
    params: AnnoMassParams,
    progress_cb: Callable[[str], None] = lambda msg: None,
    should_stop: Callable[[], bool] = lambda: False,
):
    """Run the staged AnnoMASS pipeline on positive-ion spectra.

    Returns (result_df, mf_cache, empty DataFrame).  ``pn``, ``deepmass_neg``
    and ``ref_negative`` are accepted for caller compatibility and unused.
    Raises RuntimeError on failure or when should_stop() becomes true.
    """
    def _check_stop():
        if should_stop():
            raise RuntimeError("Stopped by user.")

    positive = [i for i, sp in enumerate(spectrums) if not _is_negative_spectrum(sp)]
    n_negative = len(spectrums) - len(positive)
    if n_negative:
        progress_cb(f"{n_negative} negative-ion spectra skipped (AnnoMASS annotates positive ion only).")
    if not positive:
        raise RuntimeError("No positive-ion spectra to annotate.")

    # ── Step 1: DeepMASS structure candidates (also the flavonoid gate) ──
    progress_cb("Step 1/5: DeepMASS structure-candidate retrieval...")
    identified = {}

    def _tol(sp):
        return precursor_tolerance_ppm(sp.get("precursor_mz"), params.mz_tol_ppm)

    def _id_one(sp):
        sp.set("only_flavonoids", True)
        sp.set("candidate_ppm", _tol(sp))
        sp.set("flav_classes", sorted(STRICT_FLAV_CLASSES))
        return identify_unknown(sp, pp, deepmass_pos, ref_positive, default_database,
                                candidate_ppm=params.mz_tol_ppm)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_id_one, spectrums[i]): i for i in positive}
        for done, fut in enumerate(as_completed(futs), start=1):
            _check_stop()
            i = futs[fut]
            try:
                identified[i] = fut.result()
            except Exception:
                identified[i] = spectrums[i]
            progress_cb(f"Step 1/5: Identifying {done}/{len(positive)}")
    _check_stop()

    _min_frag = int(params.min_fragment_peaks or 0)
    _frag_ok = {i for i, sp in identified.items()
                if _min_frag <= 0 or _effective_fragment_count(sp) >= _min_frag}
    all_ms2 = [i for i in positive if _ms2_peak_count(identified[i]) > 0]
    flav = [i for i in all_ms2 if identified[i].get("is_flavonoid")]
    # The validated flavonoid pre-filter decides which spectra enter the library
    # levels (A/B/C); --benchmark-all-ms2 forwards every MS2 spectrum.
    if params.benchmark_all_ms2 or not flav:
        flav = list(all_ms2)
    if not flav:
        raise RuntimeError("No spectra with MS/MS peaks found.")
    progress_cb(f"Step 2/5: {len(flav)} spectra enter the library levels (A -> B -> C).")

    rows = []

    def _new_row(**kw):
        kw["_pipeline_uid"] = len(rows)
        rows.append(kw)
        return kw

    def _neutral(sp):
        try:
            prec = float(sp.get("precursor_mz") or 0.0)
        except Exception:
            prec = 0.0
        from core.importing.load_from_files import _neutral_mass_from_adduct
        return prec, _neutral_mass_from_adduct(prec, _spectrum_adduct(sp))

    def _deepmass_lookup(sp):
        """InChIKey14 -> (Consensus, DeepMASS_raw, Formula, Structure score)."""
        out = {}
        annot = sp.metadata.get("annotation", None)
        if annot is None or getattr(annot, "empty", True):
            return out
        for _, r in annot.iterrows():
            ik = _norm_ikey(r.get("InChIKey", ""))[:14]
            if not ik:
                continue
            cs = float(r.get("Consensus Score", 0.0) or 0.0)
            if ik not in out or cs > out[ik][0]:
                out[ik] = (cs, float(r.get("DeepMASS_raw", 0.0) or 0.0),
                           r.get("Formula Score"), r.get("Structure Score"))
        return out

    def _base_row(sp, i, stage):
        prec, _ = _neutral(sp)
        return dict(stage=stage, spectrum_index=i, spectrum_title=_spectrum_title(sp, i),
                    precursor_mz=prec, query_adduct=_spectrum_adduct(sp),
                    ms2_peaks=_ms2_peak_count(sp), mf_adduct=_spectrum_adduct(sp) or params.mf_adduct)

    modules = _levels.parse_modules(params.enabled_levels)
    if modules != "ABCD":
        progress_cb("Modules selected: " + ", ".join(f"{m} ({_levels.MODULE_NAMES[m]})" for m in modules))

    # ── Level A: DreaMS library retrieval ─────────────────────────────
    _check_stop()
    if "A" in modules and not _ensure_dreams_adapter():
        raise RuntimeError(f"DreaMS is not available: {_DREAMS_IMPORT_ERROR or 'import failed'}")
    progress_cb("Step 3/5: Level A -- DreaMS library retrieval..." if "A" in modules
                else "Step 3/5: Level A (DreaMS) not selected.")
    tmp_mgf = os.path.join(tempfile.gettempdir(), f"annomass_dreams_query_{os.getpid()}.mgf")
    written, local_idx = 0, []
    if "A" in modules:
        written, local_idx, _titles = dreams_spectra_to_mgf(
            [identified[i] for i in flav], tmp_mgf, return_mapping=True)
    q_to_orig = {q: flav[l] for q, l in enumerate(local_idx) if l < len(flav)}
    dreams_df = pd.DataFrame(columns=["query_index"])
    if written and "A" in modules:
        dreams_df = run_dreams_search(
            mgf_path=tmp_mgf, ref_hdf5=ref_hdf5, top_n=params.dreams_top_n,
            cho_only=params.cho_only, mz_tol_ppm=params.mz_tol_ppm,
            mz_tol_ppm_per_query=[_tol(identified[flav[l]]) for l in local_idx],
            batch_size=params.dreams_batch_size,
            progress_callback=lambda msg: progress_cb(f"Step 3/5: {msg}"),
        )
        if dreams_df is None:
            dreams_df = pd.DataFrame(columns=["query_index"])

    deepmass_by_spec = {}
    for qi, hits in (dreams_df.groupby("query_index") if not dreams_df.empty else []):
        orig_i = q_to_orig.get(int(qi))
        if orig_i is None:
            continue
        sp = identified[orig_i]
        _, neutral = _neutral(sp)
        dm = deepmass_by_spec.setdefault(orig_i, _deepmass_lookup(sp))
        first = hits.iloc[0]
        a_hits = []
        for _, h in hits.iterrows():
            smi = _norm_smi(h.get("library_SMILES", "") or h.get("library_smiles", ""))
            classes = _strict_classes(smi)
            if not classes:
                continue
            formula = _norm_formula(h.get("library_formula", ""))
            mass = _formula_mass(formula)
            ppm = _ppm_error(mass, neutral) if mass is not None and neutral else None
            if ppm is not None and abs(ppm) > _tol(sp):
                continue
            ik = _norm_ikey(h.get("library_inchikey", ""))
            dmv = dm.get(ik[:14], (0.0, 0.0, None, None))
            row = _base_row(sp, orig_i, "A")
            row.update(
                compound_name=_clean_text(h.get("library_name", "")) or ik or smi,
                formula=formula, smiles=smi, inchikey=ik,
                flavonoid_class=";".join(sorted(classes)), ms1_error_ppm=ppm,
                DreaMS_similarity=float(h.get("DreaMS_similarity", 0.0) or 0.0),
                DreaMS_topk=int(h.get("topk", 0) or 0),
                Modified_cosine_similarity=float(h.get("Modified_cosine_similarity", 0.0) or 0.0),
                library_cosine=float(h.get("Modified_cosine_similarity", 0.0) or 0.0),
                library_peaks=str(h.get("library_peaks", "") or ""),
                library_ID=str(h.get("library_ID", "") or ""),
                deepmass_score=dmv[0], DeepMASS_raw=dmv[1],
                candidate_source="database", source="DreaMS",
                dreams_umap_x=first.get("dreams_umap_x"), dreams_umap_y=first.get("dreams_umap_y"),
                dreams_cluster=first.get("dreams_cluster"),
            )
            a_hits.append(row)
        # one A candidate per compound: its library spectra are pooled
        for row in _levels.aggregate_dreams_hits(a_hits, params.level_a_weight):
            _new_row(**row)
    a_frame = pd.DataFrame([r for r in rows if r["stage"] == "A"])
    annotated = _annotated_spectra(a_frame, params)
    progress_cb(f"Level A: {len(annotated)}/{len(flav)} spectra annotated by DreaMS.")

    # ── Levels B / C: candidates carried down from A + DeepMASS candidates ─────
    # Spectra with no passing A.  The candidate pool is the DeepMASS structure
    # candidates plus the DreaMS hits that failed A; each candidate carries its
    # DreaMS similarity and best direct library cosine (DreaMS library hit or
    # DeepMASS experimental library spectrum) into B, and its B score into C.
    _check_stop()
    exclude = ()
    if params.ref_pos_exclude:
        exclude = np.load(params.ref_pos_exclude).tolist()
    lib_index = LibrarySpectrumIndex(ref_positive, exclude=exclude,
                                     per_compound=params.library_spectra_per_compound)
    a_by_spec = defaultdict(list)
    for r in rows:
        if r["stage"] == "A":
            a_by_spec[r["spectrum_index"]].append(r)
    bc_spectra = [i for i in flav if params.evaluate_all_stages or i not in annotated]
    if "B" not in modules and "C" not in modules:
        bc_spectra = []
    progress_cb(f"Step 4/5: Levels B/C -- {len(bc_spectra)} spectra go to the B/C candidate pool.")
    c_rows = []
    _carry = ("DreaMS_similarity", "DreaMS_topk", "Modified_cosine_similarity")

    def _emit(row, has_library):
        if has_library and "B" in modules:
            _new_row(**dict(row, stage="B"))
        if "C" in modules:
            c_rows.append(dict(row, stage="C"))

    for n_done, i in enumerate(bc_spectra, start=1):
        _check_stop()
        sp = identified[i]
        _, neutral = _neutral(sp)
        # DreaMS hits of this spectrum (they all failed A), best scores per InChIKey14
        dreams_k14 = {}
        for r in a_by_spec.get(i, []):
            k = str(r.get("inchikey") or "")[:14] or r.get("smiles")
            cur = dreams_k14.get(k)
            if cur is None:
                dreams_k14[k] = dict(r, _best_cos=r["library_cosine"],
                                     _best_peaks=r.get("library_peaks", ""))
            else:
                prev_cos, prev_peaks = cur["_best_cos"], cur.get("_best_peaks", "")
                if r["library_cosine"] > prev_cos:
                    prev_cos, prev_peaks = r["library_cosine"], r.get("library_peaks", "")
                if r["DreaMS_similarity"] > cur["DreaMS_similarity"]:
                    dreams_k14[k] = dict(r, _best_cos=prev_cos, _best_peaks=prev_peaks)
                else:
                    cur["_best_cos"], cur["_best_peaks"] = prev_cos, prev_peaks
        covered = set()
        annot = sp.metadata.get("annotation", None)
        if annot is not None and not getattr(annot, "empty", True):
            annot = annot.sort_values("Consensus Score", ascending=False)
            if params.deepmass_candidate_top_n > 0:
                annot = annot.head(params.deepmass_candidate_top_n)
            seen = set()
            for _, c in annot.iterrows():
                smi = _norm_smi(c.get("CanonicalSMILES", ""))
                classes = _strict_classes(smi)
                if not classes:
                    continue
                ik = _norm_ikey(c.get("InChIKey", ""))
                key = smi or ik
                if key in seen:
                    continue
                seen.add(key)
                formula = _norm_formula(c.get("MolecularFormula", ""))
                mass = _formula_mass(formula)
                ppm = _ppm_error(mass, neutral) if mass is not None and neutral else None
                if ppm is not None and abs(ppm) > _tol(sp):
                    continue
                name = _clean_text(c.get("Title", "")) or _clean_text(c.get("compound_name", "")) or ik
                k14 = ik[:14] or smi
                d = dreams_k14.get(k14)
                covered.add(k14)
                lib_cos, n_ref, lib_peaks = (lib_index.best_match(sp, ik) if lib_index.has(ik)
                                             else (0.0, 0, ""))
                if d and float(d["_best_cos"]) > lib_cos:
                    lib_peaks = d.get("_best_peaks", "")
                row = _base_row(sp, i, "B")
                row.update(
                    compound_name=name, formula=formula, smiles=smi, inchikey=ik,
                    flavonoid_class=";".join(sorted(classes)), ms1_error_ppm=ppm,
                    deepmass_score=float(c.get("Consensus Score", 0.0) or 0.0),
                    DeepMASS_raw=float(c.get("DeepMASS_raw", 0.0) or 0.0),
                    DreaMS_similarity=float(d["DreaMS_similarity"]) if d else 0.0,
                    DreaMS_topk=d.get("DreaMS_topk") if d else None,
                    Modified_cosine_similarity=float(d["_best_cos"]) if d else 0.0,
                    library_cosine=max(lib_cos, float(d["_best_cos"]) if d else 0.0),
                    library_reference_count=n_ref,
                    library_peaks=lib_peaks,
                    candidate_source="database",
                    source="DeepMASS" + ("+DreaMS" if d else "") + ("+library spectrum" if n_ref else ""),
                )
                for fc in ("Formula Score", "Structure Score", "Consensus Score"):
                    if fc in c:
                        row[fc] = c.get(fc)
                _emit(row, has_library=bool(d) or n_ref > 0)
        # DreaMS hits whose structure is not among the DeepMASS candidates
        for k14, d in dreams_k14.items():
            if k14 in covered:
                continue
            lib_cos, n_ref, lib_peaks = (lib_index.best_match(sp, d["inchikey"]) if lib_index.has(d["inchikey"])
                                         else (0.0, 0, ""))
            if d["_best_cos"] >= lib_cos:
                lib_peaks = d.get("_best_peaks", "")
            row = {k: v for k, v in d.items() if k not in ("_pipeline_uid", "_best_cos", "_best_peaks")}
            row.update(library_cosine=max(lib_cos, d["_best_cos"]), library_reference_count=n_ref,
                       library_peaks=lib_peaks,
                       Modified_cosine_similarity=d["_best_cos"], source="DreaMS (failed A)")
            _emit(row, has_library=True)
        if n_done % 50 == 0:
            progress_cb(f"Step 4/5: B/C candidates {n_done}/{len(bc_spectra)}")

    b_frame = pd.DataFrame([r for r in rows if r["stage"] == "B"])
    annotated_b = _annotated_spectra(b_frame, params)
    progress_cb(f"Level B: {len(annotated_b - annotated)} more spectra annotated with library evidence.")
    # ICEBERG only where a C result can still matter
    for row in c_rows:
        if params.evaluate_all_stages or row["spectrum_index"] not in annotated_b:
            _new_row(**row)
    mf_cache_by_uid = {}
    c_frame = pd.DataFrame([r for r in rows if r["stage"] == "C"])
    if not c_frame.empty:
        mf_cache_by_uid.update(_score_with_iceberg(
            c_frame, identified, params, progress_cb, _check_stop, "Level C"))
        for col in ("mf_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity"):
            for uid, v in zip(c_frame["_pipeline_uid"], c_frame[col]):
                rows[int(uid)][col] = float(v)
    annotated_c = _annotated_spectra(c_frame, params)
    annotated_abc = annotated | annotated_b | annotated_c
    progress_cb(f"Level C: {len(annotated_c - annotated - annotated_b)} more spectra annotated "
                f"by DeepMASS + predicted spectrum.")

    # ── Level D: DeepMet CLM library-external structures ─────────────────
    if params.use_deepmet_clm and "D" in modules:
        # D runs on every input spectrum, in parallel with A/B/C: its result is an EXTRA output that never changes the A/B/C ranking.
        d_spectra = list(all_ms2)
        progress_cb(f"Step 5/5: Level D -- DeepMet CLM on {len(d_spectra)} spectra (extra output; A/B/C are not affected)...")
        try:
            from core import deepmet_clm_adapter
            db_keys = deepmet_clm_adapter.database_structure_keys(default_database)
            db_formula_counts = _structure_library_formula_counts(default_database)
            d_start = len(rows)
            for i in d_spectra:
                _check_stop()
                sp = identified[i]
                if _effective_fragment_count(sp) < _MIN_EFFECTIVE_FRAGMENTS:
                    continue
                prec, neutral = _neutral(sp)
                if sp.get("neutral_mass") is not None:
                    neutral = sp.get("neutral_mass")
                if neutral is None:
                    continue
                cand = deepmet_clm_adapter.match_neutral_mass(
                    neutral, database_keys=db_keys,
                    ppm_tolerance=params.deepmet_clm_mass_tol_ppm,
                    top_n=params.deepmet_clm_top_n,
                )
                seen = set()
                for _, c in cand.iterrows():
                    smi = _norm_smi(c.get("canonical_smiles", ""))
                    if not smi or smi in seen:
                        continue
                    classes = _strict_classes(smi)
                    if not classes or not _is_plausible_denovo_structure(smi):
                        continue
                    seen.add(smi)
                    ik = deepmet_clm_adapter.inchikey_from_smiles(smi)
                    formula = _formula_key(c.get("molecular_formula", ""))
                    n_formula = int(db_formula_counts.get(formula, 0))
                    row = _base_row(sp, i, "D")
                    row.update(
                        compound_name=ik or smi, formula=formula, smiles=smi, inchikey=ik,
                        flavonoid_class=";".join(sorted(classes)),
                        ms1_error_ppm=float(c.get("mass_error_ppm", 0.0)),
                        formula_in_structure_library=n_formula > 0,
                        structure_library_formula_count=n_formula,
                        deepmass_score=0.0, candidate_source="deepmet_clm", source="DeepMet CLM",
                        deepmet_sampling_frequency=float(c.get("sampling_frequency", 0.0)),
                    )
                    _new_row(**row)
            d_frame = pd.DataFrame(rows[d_start:])
            if not d_frame.empty:
                mf_cache_by_uid.update(_score_with_iceberg(
                    d_frame, identified, params, progress_cb, _check_stop, "Level D"))
                for col in ("mf_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity"):
                    for uid, v in zip(d_frame["_pipeline_uid"], d_frame[col]):
                        rows[int(uid)][col] = float(v)
        except Exception as _clm_exc:
            progress_cb(f"Level D skipped (DeepMet CLM error: {_clm_exc}).")
    else:
        progress_cb("Step 5/5: Level D (DeepMet CLM) " + ("not selected." if "D" not in modules else "disabled."))

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        progress_cb("No candidate at any level.")
        return pd.DataFrame(), {}, pd.DataFrame()
    for col in ("DreaMS_similarity", "Modified_cosine_similarity", "library_cosine",
                "deepmass_score", "DeepMASS_raw", "mf_score",
                "iceberg_entropy_similarity", "iceberg_cosine_similarity",
                "deepmet_sampling_frequency"):
        if col not in candidates.columns:
            candidates[col] = 0.0
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0.0)

    # Calibration / audit hook: the complete candidate table of every stage,
    # before the waterfall picks the reported level.
    stage_dump = os.environ.get("ANNOMASS_DUMP_STAGE_CANDIDATES", "").strip()
    if stage_dump:
        candidates.to_csv(stage_dump, index=False, encoding="utf-8-sig")
        progress_cb(f"Stage candidate table written to {stage_dump}.")

    # ── Waterfall A -> B -> C: first passing level per spectrum.  Level D is evaluated on its own
    #    (every spectrum) and only appended after the A/B/C table is final, so it never changes A/B/C. ──
    is_d = candidates["stage"].eq("D")
    result = _levels.assign_levels(candidates[~is_d], params, identity=_row_identity)
    result_d = _levels.assign_levels(candidates[is_d], params, identity=_row_identity)
    _all_levels = pd.concat([result, result_d], ignore_index=True)
    level_counts = _all_levels.drop_duplicates(["spectrum_index", "confidence_level"])["confidence_level"].value_counts().to_dict() \
        if not _all_levels.empty else {}
    n_unannotated = len(set(all_ms2) - set(_all_levels["spectrum_index"])) if not _all_levels.empty else len(all_ms2)
    progress_cb(f"Annotation levels per spectrum (D is an extra output, counted separately): {level_counts}; "
                f"{n_unannotated} spectra without any passing level.")
    if result.empty and result_d.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    def _postprocess(res, topk):
        if res.empty:
            return res
        # AnnoSM gated rerank inside each spectrum (same level only)
        if params.enable_annosm_rerank:
            res = _annosm_rerank(res, identified, params, progress_cb, _check_stop)
        # compound dedup / support aggregation
        res = _aggregate_compounds(res, params, _frag_ok, _check_stop)
        if res.empty:
            return res
        _add_fragment_evidence(res, identified, params)
        if topk and params.top_n_per_spectrum > 0:
            res = _select_topk_per_spectrum(
                res, params.top_n_per_spectrum, order_col="_rank_key", ascending=False
            ).sort_values(["_rank_key", "support_spectra_count"], ascending=[False, False],
                          kind="stable").reset_index(drop=True)
            progress_cb(f"Final Top-{params.top_n_per_spectrum}: {len(res)} rows across "
                        f"{res['spectrum_index'].nunique()} spectra.")
        return res

    result = _postprocess(result, topk=True)
    result_d = _postprocess(result_d, topk=False)     # dual Top-1: one row per spectrum, outside the Top-K quota
    if not result_d.empty:
        progress_cb(f"Level D (extra output): {len(result_d)} rows across {result_d['spectrum_index'].nunique()} spectra.")
    result = pd.concat([result, result_d], ignore_index=True) if not result.empty else result_d
    if result.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    # ── ABCoRT predicted retention time (informational) ─────────────────
    if params.enable_abcort_predict and "smiles" in result.columns:
        _add_abcort_rt(result, params, progress_cb)

    # which modules produced this table (module selection, see core/staged_levels.py)
    result["modules_run"] = modules if params.use_deepmet_clm else modules.replace("D", "")
    return _finalise_output(result, mf_cache_by_uid, progress_cb)


def _annosm_rerank(result, identified, params, progress_cb, check_stop):
    try:
        from core import annosm_bridge as _ab
    except Exception:
        progress_cb("AnnoSM rerank skipped (import failed).")
        return result
    if not _ab.is_available():
        progress_cb(f"AnnoSM rerank skipped ({_ab.unavailable_reason()}).")
        return result
    progress_cb("Step 6: AnnoSM evidence rerank (extended library)...")
    try:
        preset = _ab.ANNOSM_LIBRARIES[_ab.PIPELINE_RERANK_PRESET]
        screener = _ab.get_screener(preset["tpil"])
        annotator = _ab.get_annotator(preset["cfil"])
    except Exception as exc:
        progress_cb(f"AnnoSM rerank skipped ({exc}).")
        return result
    external = result["candidate_source"].eq("deepmet_clm") & (not params.annosm_include_external)
    bypass = result[external]
    groups, n_promoted = [], 0
    for sp_idx, group in result[~external].groupby("spectrum_index", sort=False):
        check_stop()
        sp = identified.get(sp_idx)
        try:
            peaks = np.column_stack([np.asarray(sp.peaks.mz, float),
                                     np.asarray(sp.peaks.intensities, float)])
            prec = float(sp.get("precursor_mz", ""))
        except Exception:
            groups.append(group)
            continue
        if len(group) < 2 or len(peaks) == 0:
            groups.append(group)
            continue
        try:
            reranked = _ab.gated_rerank_spectrum_group(group, prec, peaks, screener, annotator,
                                                       max_level_score_gap=params.annosm_max_level_gap)
        except Exception:
            groups.append(group)
            continue
        if bool(reranked.get("annosm_promoted", pd.Series(dtype=bool)).any()):
            n_promoted += 1
        groups.append(reranked)
    progress_cb(f"Step 6: AnnoSM rerank done, {n_promoted} spectra promoted a challenger.")
    return pd.concat(groups + [bypass], ignore_index=True)


def _aggregate_compounds(result, params, frag_ok, check_stop):
    """One row per compound (per spectrum with dedup_per_spectrum); support
    columns list every spectrum that reported the compound."""
    def _key(r):
        return _candidate_route_dedup_identity(r, per_spectrum=params.dedup_per_spectrum)

    def _sid(title, fallback):
        m = re.search(r"(?:^|\|)ID=([^|]+)", str(title or ""))
        return m.group(1).strip() if m else str(fallback)

    min_frag = int(params.min_fragment_peaks or 0)
    reps = []
    for _, g in result.groupby(result.apply(_key, axis=1), sort=False):
        check_stop()
        g = g.sort_values("_rank_key", ascending=False)
        if min_frag > 0 and not g["spectrum_index"].isin(frag_ok).any():
            continue
        rep = g.iloc[0].copy()
        spec_ids = list(dict.fromkeys(g["spectrum_index"].tolist()))
        titles = list(dict.fromkeys(str(t) for t in g["spectrum_title"] if str(t)))
        rep["support_spectra_count"] = len(spec_ids)
        rep["spectrum_ids"] = ";".join(dict.fromkeys(
            _sid(t, s) for t, s in zip(g["spectrum_title"], g["spectrum_index"])))
        rep["support_spectrum_indices"] = ";".join(str(x) for x in spec_ids[:50])
        rep["support_spectrum_titles"] = "; ".join(titles[:10])
        rep["support_precursor_mz"] = ";".join(dict.fromkeys(str(x) for x in g["precursor_mz"]))
        rep["support_spectrum_levels"] = ";".join(
            f"{s}:{l}" for s, l in dict.fromkeys(zip(g["spectrum_index"], g["confidence_level"])))
        reps.append(rep)
    if not reps:
        return pd.DataFrame()
    out = pd.DataFrame(reps)
    return out.sort_values(["_rank_key", "support_spectra_count"],
                           ascending=[False, False], kind="stable").reset_index(drop=True)


def _add_fragment_evidence(result, identified, params):
    """Intensity-weighted share of the query peaks explained by fixed losses /
    RDA / combined losses.  Column + manual-review flag only."""
    if params.evidence_score_flag_threshold <= 0 or "smiles" not in result.columns:
        return
    try:
        table = _load_fragment_loss_table()
    except Exception:
        return
    scores, flags = [], []
    for _, r in result.iterrows():
        try:
            sp = identified[int(r["spectrum_index"])]
            peaks = list(zip(np.asarray(sp.peaks.mz, float).tolist(),
                             np.asarray(sp.peaks.intensities, float).tolist()))
            s = float(_score_fragment_evidence(str(r.get("smiles") or ""),
                                               float(sp.get("precursor_mz") or 0.0),
                                               peaks, table).evidence_score)
            scores.append(s)
            flags.append("low fragment support -- manual review recommended"
                         if s < params.evidence_score_flag_threshold else "")
        except Exception:
            scores.append(None)
            flags.append("")
    result["evidence_score"] = scores
    result["evidence_flag"] = flags


def _add_abcort_rt(result, params, progress_cb):
    try:
        from core import abcort_bridge as _abc
    except Exception:
        progress_cb("ABCoRT prediction skipped (import failed).")
        return
    if not _abc.is_available():
        progress_cb(f"ABCoRT prediction skipped ({_abc.unavailable_reason()}).")
        return
    progress_cb("ABCoRT retention-time prediction...")
    try:
        preds = _abc.predict_rt_batch(result["smiles"].fillna("").tolist(),
                                      checkpoint_path=params.abcort_checkpoint_path)
        result["predicted_rt_min"] = [p["predicted_rt_min"] if p is not None else None for p in preds]
        progress_cb(f"ABCoRT RT predicted for "
                    f"{sum(1 for p in preds if p is not None and p['valid'])}/{len(preds)} compounds.")
    except Exception as exc:
        progress_cb(f"ABCoRT prediction skipped ({exc}).")


_PREFERRED_COLUMNS = [
    "spectrum_title", "spectrum_index", "spectrum_rank",
    "compound_name", "rank", "confidence_level", "msi_level", "level_score",
    "evidence_level", "modules_run", "formula", "smiles", "inchikey", "flavonoid_class",
    "DreaMS_similarity", "library_cosine", "deepmass_score",
    "spectrum_predictor_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity",
    "DreaMS_topk", "dreams_hit_count", "Modified_cosine_similarity", "library_reference_count", "DeepMASS_raw",
    "annosm_state", "annosm_score", "annosm_promoted",
    "predicted_rt_min",
    "support_spectra_count", "spectrum_ids", "support_precursor_mz",
    "support_spectrum_indices", "support_spectrum_levels", "support_spectrum_titles",
    "query_adduct", "ms1_error_ppm", "ms2_peaks", "precursor_mz",
    "formula_in_structure_library", "structure_library_formula_count",
    "candidate_source", "deepmet_sampling_frequency", "source",
    "spectrum_predictor_adduct", "evidence_score", "evidence_flag",
]


def _finalise_output(result, mf_cache_by_uid, progress_cb):
    mf_cache = {}
    for row_idx, uid in enumerate(result["_pipeline_uid"].tolist()):
        try:
            uid = int(uid)
        except Exception:
            continue
        if uid in mf_cache_by_uid:
            mf_cache[row_idx] = mf_cache_by_uid[uid]
    result = result.drop(columns=["_pipeline_uid", "_rank_key", "stage"], errors="ignore")
    result.insert(1, "rank", range(1, len(result) + 1))
    result.insert(2, "spectrum_rank", result.groupby("spectrum_index", sort=False).cumcount() + 1)

    # Always recompute the InChIKey from the row's own SMILES (upstream keys
    # are frequently flat/no-stereo even when the SMILES carries stereo).
    def _ikey(row):
        smi = _norm_smi(row.get("smiles", ""))
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ik = Chem.MolToInchiKey(mol)
                if ik:
                    return ik.upper()
        return _norm_ikey(row.get("inchikey", ""))
    result["inchikey"] = result.apply(_ikey, axis=1)
    result = result.rename(columns={"mf_score": "spectrum_predictor_score",
                                    "mf_adduct": "spectrum_predictor_adduct"})
    # Measured-spectrum levels carry no predicted spectrum.
    for col in ("spectrum_predictor_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity"):
        if col in result.columns:
            result.loc[result["confidence_level"].isin(["A", "B"]), col] = np.nan
    result = result[[c for c in _PREFERRED_COLUMNS if c in result.columns]
                    + [c for c in result.columns if c not in _PREFERRED_COLUMNS]]
    progress_cb("Done! Top candidate: "
                + str(result.iloc[0]["compound_name"] if len(result) else "none"))
    return result, mf_cache, pd.DataFrame()
