"""Bridge to the standalone AnnoSM project (py_annotator) for the AnnoMASS GUI.

AnnoSM (substituent-mode annotation for flavonoids) lives in a sibling
project directory, not inside this repo:

    <ANNOMASS_HOME>/external/AnnoSM/py_annotator/   -- Python port of the R Shiny app
    <ANNOMASS_HOME>/external/AnnoSM/Library/        -- TPIL/CFIL library files

This module is the single place that knows that absolute path, so the rest
of the GUI never has to. If the AnnoSM project or py_annotator is missing on
a given machine, `is_available()` returns False and callers are expected to
degrade gracefully (disable the relevant UI, skip the pipeline rerank step)
rather than fail.

Two library presets are available, matching AnnoSM/REPORT_2026-07-13.md:
  - "original": the author's original two files (CFIL.xlsx + TPIL.xlsx),
    validated against the R implementation on 148 standards (<1e-8 error).
  - "extended": the extended library (CFIL_extended.csv + TPIL_extended.xlsx)
    used by the gated rerank that moves AnnoMASS Top1 651 -> 656 / 791.
"""

import re
import sys
from pathlib import Path

import pandas as pd

from core.staged_levels import level_rank

from core.paths import external_dir as _external_dir

ANNOSM_PROJECT_ROOT = _external_dir("AnnoSM")

ANNOSM_LIBRARIES = {
    "original": {
        "label": "Original",
        "tpil": ANNOSM_PROJECT_ROOT / "Library" / "TPIL.xlsx",
        "cfil": ANNOSM_PROJECT_ROOT / "Library" / "CFIL.xlsx",
    },
    "extended": {
        "label": "Extended",
        "tpil": ANNOSM_PROJECT_ROOT / "Library" / "TPIL_extended.xlsx",
        "cfil": ANNOSM_PROJECT_ROOT / "Library" / "CFIL_extended.csv",
    },
}

# The pipeline's one-click rerank always uses the extended library -- it's
# the one the 656/791 result was measured with (see plan/report).
PIPELINE_RERANK_PRESET = "extended"

_IMPORT_ERROR = None
if ANNOSM_PROJECT_ROOT.exists() and str(ANNOSM_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ANNOSM_PROJECT_ROOT))

try:
    from py_annotator.mgf_parser import read_mgf  # noqa: F401
    from py_annotator.screener import CompoundScreener
    from py_annotator.sm_annotator import SMAnnotator
    from py_annotator.candidate_sm import extract_candidate_sm
except Exception as exc:  # pragma: no cover - environment dependent
    CompoundScreener = None
    SMAnnotator = None
    extract_candidate_sm = None
    read_mgf = None
    _IMPORT_ERROR = exc


def is_available() -> bool:
    """Whether the AnnoSM project + py_annotator package can be used."""
    return _IMPORT_ERROR is None and ANNOSM_PROJECT_ROOT.exists()


def unavailable_reason() -> str:
    if not ANNOSM_PROJECT_ROOT.exists():
        return f"AnnoSM project not found: {ANNOSM_PROJECT_ROOT}"
    if _IMPORT_ERROR is not None:
        return f"py_annotator import failed: {_IMPORT_ERROR}"
    return ""


def library_files_exist(preset_key: str) -> bool:
    preset = ANNOSM_LIBRARIES.get(preset_key)
    if not preset:
        return False
    return Path(preset["tpil"]).exists() and Path(preset["cfil"]).exists()


_SCREENER_CACHE = {}
_ANNOTATOR_CACHE = {}


def get_screener(tpil_path):
    """Cached CompoundScreener singleton, keyed by resolved TPIL path."""
    key = str(Path(tpil_path).resolve())
    if key not in _SCREENER_CACHE:
        _SCREENER_CACHE[key] = CompoundScreener(tpil_path)
    return _SCREENER_CACHE[key]


def get_annotator(cfil_path):
    """Cached SMAnnotator singleton, keyed by resolved CFIL path.

    CFIL_extended.csv is ~135MB -- this must not be reloaded per pipeline
    run or per spectrum, only once per process.
    """
    key = str(Path(cfil_path).resolve())
    if key not in _ANNOTATOR_CACHE:
        _ANNOTATOR_CACHE[key] = SMAnnotator(cfil_path)
    return _ANNOTATOR_CACHE[key]


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_formula(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return re.sub(r"\s+", "", str(value)).rstrip("+-")


def _candidate_formula(smiles):
    """Recompute a neutral molecular formula from SMILES via RDKit.

    Ported from full_annosm_candidate_concentration.py::_candidate_formula
    so formula comparisons match the validated 656/791 benchmark config.
    """
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)) or not str(smiles).strip():
        return None
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        pass
    return _normalize_formula(rdMolDescriptors.CalcMolFormula(mol))


_COVERED_STATES = {"fully_consistent", "sm_consistent_unclassified"}


def _candidate_evidence(
    annotator, peaks, screened, candidate_sm, formula,
    result_cache, fragment_ppm, score_column, specificity_weight,
):
    """Evaluate one candidate's AnnoSM/CFIL evidence against a screened TPIL group.

    Direct port of full_annosm_candidate_concentration.py::_paper_candidate_evidence
    (minus the relax_tpil_composition option, which the validated 656/791 config
    does not use). Returns (state, score).
    """
    if candidate_sm is None:
        return "outside_paper_scope", 0.0
    if formula is None:
        return "missing_formula", 0.0
    if screened is None or screened.empty:
        return "tpil_no_match", 0.0

    formula_rows = screened[screened["_formula"] == formula]
    compatible = formula_rows[
        (formula_rows["OHsum"] == candidate_sm.oh_sum)
        & (formula_rows["OCH3sum"] == candidate_sm.och3_sum)
    ]
    if compatible.empty:
        return "tpil_incompatible", 0.0

    key = (candidate_sm.skeleton, candidate_sm.oh_sum, candidate_sm.och3_sum)
    if key not in result_cache:
        precursor = compatible.iloc[0].copy()
        precursor["Skeleton"] = candidate_sm.skeleton
        result_cache[key] = annotator.match_single_compound(
            compound_id="candidate",
            precursor_mz=precursor.get("Exp.m/z", 0),
            fragment_peaks=peaks,
            precursor_info=precursor,
            fragment_ppm=fragment_ppm,
        )
    result_group = result_cache[key]
    if result_group.empty:
        return "cfil_no_match", 0.0

    result_group = result_group.copy()
    if score_column == "HybridScore":
        result_group["HybridScore"] = (
            result_group["Score"] + specificity_weight * result_group["SpecificityScore"]
        )
    hit = result_group[result_group["Substitution modes"].isin(
        [candidate_sm.annotated_sm, candidate_sm.raw_sm]
    )]
    if hit.empty:
        return "cfil_incompatible", 0.0

    row = hit.sort_values(score_column, ascending=False).iloc[0]
    observed_class = str(row.get("Class", "") or "")
    expected_class = (
        "ISO" if candidate_sm.skeleton in {"IsoFlavones", "Isoflavanone", "Isoflavan"}
        else "FLA"
    )
    if observed_class == expected_class:
        state = "fully_consistent"
    elif observed_class in ("", "nan"):
        state = "sm_consistent_unclassified"
    else:
        state = "class_inconsistent"
    return state, float(row[score_column])


def candidate_sm_from_smiles(smiles):
    """extract_candidate_sm(smiles), guarded for a missing SMILES or AnnoSM
    being unavailable -- returns None if either the SMILES is empty or the
    structure falls outside AnnoSM's covered flavonoid skeletons (same
    guard gated_rerank_spectrum_group already applies inline)."""
    if not is_available() or not smiles:
        return None
    try:
        return extract_candidate_sm(smiles)
    except Exception:
        return None


def diagnostic_fragment_detail(
    annotator, peaks, substitution_modes, oh_sum, och3_sum,
    fragment_ppm=50.0, top_n=8,
):
    """On-demand lookup of WHICH CFIL diagnostic fragments in a query
    spectrum matched a specific candidate's substitution mode(s), with
    per-fragment m/z + intensity + the CFIL library's own textual
    annotation -- the detail match_single_compound() aggregates away into a
    single Score. Used by compare_isomers (core/agent_tools.py) to make an
    annosm_score legible down to the actual supporting ion(s) instead of
    just a number.

    Read-only re-derivation, not a modification of the third-party AnnoSM
    package: only reads `annotator.cfil` (a public, already-loaded
    attribute) and reuses its public `calc_ppm`, so the ppm math matches
    exactly what AnnoSM itself used to compute the score.

    Returns a list of {"mz", "intensity", "cfil_mz", "substitution_mode",
    "annotation"} dicts, deduplicated the same way match_single_compound
    dedupes (one abundance contribution per (matched m/z, substitution
    mode) pair), sorted by intensity descending, capped at top_n.
    """
    if annotator is None or not substitution_modes or peaks is None or len(peaks) == 0:
        return []
    cfil = getattr(annotator, "cfil", None)
    if cfil is None:
        return []
    rows = cfil[
        cfil["Substitution modes"].isin(substitution_modes)
        & (cfil["OHsum"] == oh_sum)
        & (cfil["OCH3sum"] == och3_sum)
    ]
    if rows.empty:
        return []

    seen = set()
    matches = []
    for mz_val, intensity in peaks:
        digi = round(float(mz_val), 4)
        for _, row in rows.iterrows():
            if annotator.calc_ppm(digi, row["_digit"]) <= fragment_ppm:
                key = (digi, row["Substitution modes"])
                if key in seen:
                    continue
                seen.add(key)
                annotation = row.get("Annotations")
                matches.append({
                    "mz": digi,
                    "intensity": float(intensity),
                    "cfil_mz": float(row["_digit"]),
                    "substitution_mode": row["Substitution modes"],
                    "annotation": None if pd.isna(annotation) else str(annotation),
                })
    matches.sort(key=lambda m: m["intensity"], reverse=True)
    return matches[:top_n]


def _row_level_rank(group, idx):
    for col in ("confidence_level", "stage"):
        if col in group.columns:
            return level_rank(group.at[idx, col])
    return 0


def gated_rerank_spectrum_group(
    group_df,
    precursor_mz,
    peaks,
    screener,
    annotator,
    precursor_ppm=10.0,
    fragment_ppm=50.0,
    score_column="HybridScore",
    specificity_weight=0.1,
    min_annosm_margin=0.01,
    max_deepmass_gap=0.25,
    max_level_score_gap=None,
):
    """Re-rank one spectrum's candidates using AnnoSM evidence, gated promotion only.

    `group_df` must already be sorted by the pipeline's rank_key (descending),
    i.e. row 0 is the pre-AnnoSM rank-1 candidate for this spectrum. Ported
    from full_annosm_candidate_concentration.py::_rank_group's "gated" branch
    (the config that produced the validated 656/791 result), adapted to
    operate on one live spectrum's candidates instead of an offline
    query_id batch with ground truth.

    A challenger only promotes over the existing rank-1 when ALL of:
      - both candidates have the same molecular formula
      - both have AnnoSM/CFIL evidence in a "covered" state
      - the substitution modes differ
      - annosm_score margin >= min_annosm_margin
      - deepmass_score gap <= max_deepmass_gap
      - if max_level_score_gap is set: the challenger's level_score is within
        that gap of the rank-1's (AnnoSM only breaks near-ties of the staged
        ranking, it never overturns a clear lead)
      - the challenger's annotation level (A/B/C/D, core.staged_levels) is
        >= the current rank-1's level -- CFIL fragment-consistency evidence
        breaks ties WITHIN a level and never leapfrogs a higher-trust one
        (on the MassBank strict full-InChIKey benchmark every cross-tier
        promotion, 47/47, was wrong -- AnnoSM/REPORT_2026-07-13.md).

    Returns a DataFrame with the same rows, possibly reordered, plus three
    new columns: annosm_state, annosm_score, annosm_promoted.
    """
    group = group_df.reset_index(drop=True).copy()
    group["annosm_state"] = "outside_paper_scope"
    group["annosm_score"] = 0.0
    group["annosm_promoted"] = False

    if peaks is None or len(peaks) == 0 or group.empty or precursor_mz is None:
        return group

    screened = screener.screen_single(
        precursor_mz=precursor_mz,
        fragment_peaks=peaks,
        precursor_ppm=precursor_ppm,
        frag_ppm=precursor_ppm,
        title="_annosm_live_",
    )
    if screened is None or screened.empty:
        return group
    screened = screened.copy()
    screened["_formula"] = screened["Formulas"].map(_normalize_formula)

    result_cache = {}
    modes = {}
    formulas = {}
    for idx in group.index:
        smiles = group.at[idx, "smiles"] if "smiles" in group.columns else None
        candidate_sm = extract_candidate_sm(smiles) if smiles else None
        formula = _candidate_formula(smiles) if smiles else None
        modes[idx] = candidate_sm
        formulas[idx] = formula
        try:
            state, score = _candidate_evidence(
                annotator, peaks, screened, candidate_sm, formula,
                result_cache, fragment_ppm, score_column, specificity_weight,
            )
        except Exception:
            state, score = "error", 0.0
        group.at[idx, "annosm_state"] = state
        group.at[idx, "annosm_score"] = score

    top1_idx = group.index[0]
    if group.at[top1_idx, "annosm_state"] not in _COVERED_STATES:
        return group
    top1_formula = formulas[top1_idx]
    top1_mode = modes[top1_idx]
    if top1_formula is None or top1_mode is None:
        return group
    top1_score = group.at[top1_idx, "annosm_score"]
    top1_deepmass = _safe_float(group.at[top1_idx, "deepmass_score"] if "deepmass_score" in group.columns else None)
    top1_tier = _row_level_rank(group, top1_idx)
    has_level_score = max_level_score_gap is not None and "level_score" in group.columns
    top1_level_score = _safe_float(group.at[top1_idx, "level_score"]) if has_level_score else 0.0

    challengers = []
    for idx in group.index[1:]:
        if group.at[idx, "annosm_state"] not in _COVERED_STATES:
            continue
        if formulas[idx] != top1_formula:
            continue
        row_mode = modes[idx]
        if row_mode is None or row_mode.raw_sm == top1_mode.raw_sm:
            continue
        if group.at[idx, "annosm_score"] - top1_score < min_annosm_margin:
            continue
        row_deepmass = _safe_float(group.at[idx, "deepmass_score"] if "deepmass_score" in group.columns else None)
        if top1_deepmass - row_deepmass > max_deepmass_gap:
            continue
        if has_level_score and top1_level_score - _safe_float(group.at[idx, "level_score"]) > max_level_score_gap:
            continue
        row_tier = _row_level_rank(group, idx)
        if row_tier < top1_tier:
            continue
        challengers.append(idx)

    if not challengers:
        return group

    promoted_idx = max(
        challengers,
        key=lambda i: (
            group.at[i, "annosm_score"],
            _safe_float(group.at[i, "deepmass_score"] if "deepmass_score" in group.columns else None),
        ),
    )
    group.at[promoted_idx, "annosm_promoted"] = True

    # Downstream pipeline code re-derives order from the `_rank_key` column
    # value in later stages (compound dedup, final sort), not from row
    # position. Swapping _rank_key between the promoted challenger and the
    # demoted former rank-1 makes the promotion survive those re-sorts, on
    # top of the physical reorder below (which is what the pipeline's
    # row-position-based per-spectrum top-N filter relies on).
    if "_rank_key" in group.columns:
        top1_key = group.at[top1_idx, "_rank_key"]
        promoted_key = group.at[promoted_idx, "_rank_key"]
        group.at[top1_idx, "_rank_key"] = promoted_key
        group.at[promoted_idx, "_rank_key"] = top1_key

    new_order = [promoted_idx] + [i for i in group.index if i != promoted_idx]
    return group.loc[new_order].reset_index(drop=True)
