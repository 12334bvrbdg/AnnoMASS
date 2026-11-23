# -*- coding: utf-8 -*-
"""Staged annotation levels: DreaMS -> DeepMASS -> DeepMet CLM.

Replaces the orthogonal ranking cascade.  Every spectrum walks down a fixed
waterfall and stops at the first level that produces at least one passing
candidate; only that level's candidates are reported for the spectrum.

  A  DreaMS library compound.  The DreaMS hits (library spectra) of one planar
     compound (InChIKey first block) are pooled -- a compound usually has several library spectra recorded at
     different energies / on different instruments -- and the compound scores
                                    S_A = wA * max DreaMS similarity over its hits
                                        + (1 - wA) * max modified cosine to its hit spectra
  B  spectra with no passing A.  Candidates = DeepMASS structure candidates plus
     the DreaMS hits that failed A; a candidate enters B when it has library
     evidence (a DreaMS hit or an experimental spectrum in the DeepMASS library).
     Every candidate carries its scores down:
                                    S_B = wB_M * DeepMASS score
                                        + wB_L * best library-spectrum modified cosine
                                        + wB_D * DreaMS similarity   (wB_M + wB_L + wB_D = 1)
  C  spectra with no passing B.  Every candidate (with or without library
     evidence) carries its B score down and adds the predicted spectrum:
                                    S_C = wC * S_B + (1 - wC) * ICEBERG predicted-vs-measured similarity
  D  DeepMet CLM structure absent from the structure DB: ICEBERG entropy and
     cosine both >= their thresholds and both pick the same Top-1 structure.
  -  nothing passed: the spectrum is not annotated (no row is reported).

Module selection: ``params.enabled_levels`` (default "ABCD") restricts the
waterfall to the chosen modules, e.g. "A" (DreaMS only) or "BC" (DeepMASS
only); skipped modules are not computed and can never pass.  Without A there is
no DreaMS evidence, so the B weights are renormalised over DeepMASS + library
cosine.

Within a level, candidates are ordered by the level score.  Weights and
thresholds come from AnnoMassParams; their defaults were chosen on a
benchmark-disjoint calibration set (scripts/calibration/).

MSI / Schymanski et al. (2014) confidence: A and B match an experimental
reference spectrum (Level 2), C rests on in-silico prediction plus a structure
prior (Level 3), D is a generated structure with no reference (Level 4).
Level 1 needs an in-house authentic standard and is never assigned here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LEVELS = ("A", "B", "C", "D")
LEVEL_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}
MSI_LEVEL = {"A": "Level 2", "B": "Level 2", "C": "Level 3", "D": "Level 4"}
LEVEL_DESCRIPTION = {
    "A": "Experimental spectral-library matching using learned spectral representations and peak similarity",
    "B": "Structure-library inference supported by experimental spectral similarity",
    "C": "Structure-library inference supported by predicted fragmentation similarity",
    "D": "Generated-structure inference supported by entropy and cosine agreement on the best candidate",
}
# evidence_level values written by the pipeline, one per level
EVIDENCE_LEVEL = {
    "A": "dreams_library",
    "B": "deepmass_library_spectrum",
    "C": "deepmass_predicted_spectrum",
    "D": "clm_predicted_spectrum",
}

# DeepMASS_raw = sqrt(sum over the 20 nearest references of vecsim * fpsim);
# its ceiling is sqrt(20), so this maps it onto 0..1 without the per-spectrum
# max-normalisation that the Consensus Score carries.
_DEEPMASS_RAW_CEILING = math.sqrt(20.0)


MODULE_NAMES = {
    "A": "Experimental spectral-library matching",
    "B": "Structure-library inference + experimental spectra",
    "C": "Structure-library inference + predicted fragmentation",
    "D": "Generated-structure inference",
}
_MODULE_ALIASES = {"dreams": "A", "library": "B", "deepmass": "B", "ice": "C", "iceberg": "C",
                   "predicted": "C", "clm": "D", "deepmet": "D", "all": "ABCD"}


def parse_modules(value) -> str:
    """Normalise a module selection to an ordered string over "ABCD".
    Accepts "ABCD", "A,C", "a c", ["A", "C"] or names (dreams, library, ice, clm, all)."""
    if value is None or value == "":
        return "ABCD"
    items = value if isinstance(value, (list, tuple, set)) else str(value).replace(";", ",").replace(" ", ",").split(",")
    chosen = set()
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        alias = _MODULE_ALIASES.get(item.lower())
        if alias:
            chosen.update(alias)
        elif set(item.upper()) <= set(LEVELS):
            chosen.update(item.upper())
        else:
            raise ValueError(f"unknown module {item!r}; use A/B/C/D or {', '.join(sorted(_MODULE_ALIASES))}")
    if not chosen:
        raise ValueError("select at least one module (A, B, C or D)")
    return "".join(L for L in LEVELS if L in chosen)


def enabled_levels(params) -> str:
    return parse_modules(getattr(params, "enabled_levels", "ABCD"))


def level_rank(level: object) -> int:
    """Ordinal of a level (A highest); unknown/blank ranks lowest."""
    return LEVEL_RANK.get(str(level or ""), 0)


def _num(value, default=0.0):
    try:
        v = float(value)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def aggregate_dreams_hits(hits: list, weight: float = 0.5) -> list:
    """Pool the DreaMS hits of one spectrum per compound.  The evidence is
    pooled per planar structure (InChIKey first block, else SMILES): every
    library spectrum of the compound counts, whatever stereo variant the
    library stored it under, and the compound gets its best DreaMS similarity
    and its best library cosine (the two maxima may come from different
    library spectra), plus ``dreams_hit_count``.  Each stereo variant (SMILES;
    the DreaMS library stores only the InChIKey first block) stays a separate
    candidate carrying the pooled scores.  The variants of one compound share
    the pooled scores, so they are ordered by their own direct spectral match:
    ``a_hit_cosine`` (the variant's best library cosine) then ``a_hit_dreams``
    -- e.g. a glucoside and a galactoside share the InChIKey first block, and
    the direct cosine separates them better than the embedding.
    ``a_hit_score`` is the variant's best per-hit A score (*weight* = level-A
    DreaMS weight), kept for reference."""
    def hit_score(h):
        return weight * _num(h.get("DreaMS_similarity")) + (1 - weight) * _num(h.get("library_cosine"))

    groups = {}
    for h in hits:
        key = (h.get("spectrum_index"), str(h.get("inchikey") or "")[:14] or h.get("smiles"))
        groups.setdefault(key, []).append(h)
    out = []
    for g in groups.values():
        best_d = max(_num(h.get("DreaMS_similarity")) for h in g)
        best_c = max(_num(h.get("library_cosine")) for h in g)
        variants = {}
        for h in g:
            v = h.get("smiles") or str(h.get("inchikey") or "")
            variants.setdefault(v, []).append(h)
        for vh in variants.values():
            rep = max(vh, key=hit_score)
            best_cos_hit = max(vh, key=lambda h: _num(h.get("library_cosine")))
            out.append(dict(rep, DreaMS_similarity=best_d, library_cosine=best_c,
                            library_peaks=best_cos_hit.get("library_peaks", "") or "",
                            library_spectrum_id=best_cos_hit.get("library_ID", "") or "",
                            Modified_cosine_similarity=best_c, dreams_hit_count=len(g),
                            a_hit_score=hit_score(rep),
                            a_hit_cosine=max(_num(h.get("library_cosine")) for h in vh),
                            a_hit_dreams=max(_num(h.get("DreaMS_similarity")) for h in vh)))
    return out


def deepmass_value(row, mode: str = "consensus") -> float:
    """DeepMASS evidence used in the B/C fusion.

    consensus: the DeepMASS Consensus Score (0.3 formula + 0.7 structure,
               structure part max-normalised within each spectrum).
    absolute:  DeepMASS_raw / sqrt(20), not normalised per spectrum.
    """
    if mode == "absolute":
        return min(max(_num(row.get("DeepMASS_raw")) / _DEEPMASS_RAW_CEILING, 0.0), 1.0)
    return _num(row.get("deepmass_score"))


def ice_value(row, metric: str = "mf_cosine") -> float:
    """ICEBERG predicted-vs-measured similarity used in the C fusion."""
    if metric == "ice_entropy":
        return _num(row.get("iceberg_entropy_similarity"))
    if metric == "ice_cosine":
        return _num(row.get("iceberg_cosine_similarity"))
    return _num(row.get("mf_score"))


def b_score(row, params) -> float:
    """Level-B score from the scores a candidate carries (0 for missing ones)."""
    w_m = params.level_b_weight
    w_l = params.level_b_library_weight
    w_d = max(0.0, 1.0 - w_m - w_l)
    if "A" not in enabled_levels(params) and w_m + w_l > 0:
        # DreaMS was not run: spread its weight over the remaining evidence
        w_m, w_l, w_d = w_m / (w_m + w_l), w_l / (w_m + w_l), 0.0
    return (w_m * deepmass_value(row, params.deepmass_score_mode)
            + w_l * _num(row.get("library_cosine"))
            + w_d * _num(row.get("DreaMS_similarity")))


def level_score(row, params) -> float:
    """Fused score of one candidate for its own stage.  A: DreaMS + direct
    cosine; B: DeepMASS + direct cosine + DreaMS (carried from A); C: the
    carried B score + ICEBERG.  D rows report the mean of their two ICEBERG
    similarities (display only; D is decided by the dual Top-1 rule)."""
    stage = str(row.get("stage", "") or "")
    if stage == "A":
        w = params.level_a_weight
        return w * _num(row.get("DreaMS_similarity")) + (1 - w) * _num(row.get("library_cosine"))
    if stage == "B":
        return b_score(row, params)
    if stage == "C":
        w = params.level_c_weight
        return w * b_score(row, params) + (1 - w) * ice_value(row, params.level_c_ice_metric)
    if stage == "D":
        return 0.5 * (_num(row.get("iceberg_entropy_similarity"))
                      + _num(row.get("iceberg_cosine_similarity")))
    return 0.0


def _threshold(stage: str, params) -> float:
    return {
        "A": params.level_a_threshold,
        "B": params.level_b_threshold,
        "C": params.level_c_threshold,
    }[stage]


def stereo_specificity(smiles: object) -> int:
    s = str(smiles or "")
    return int("@" in s or "/" in s or "\\" in s)


def clm_dual_top1(frame: pd.DataFrame, primary_min=0.5, secondary_min=0.4) -> pd.Series:
    """True for the one CLM row per spectrum that tops BOTH ICEBERG metrics and
    whose scores satisfy (entropy >= primary and cosine >= secondary) OR
    (cosine >= primary and entropy >= secondary) (ties broken by sampling
    frequency, then mass error)."""
    out = pd.Series(False, index=frame.index)
    if frame.empty:
        return out
    for _, group in frame.groupby("spectrum_index", sort=False):
        cols = ["deepmet_sampling_frequency", "ms1_error_ppm"]
        g = group.copy()
        for c in cols + ["iceberg_entropy_similarity", "iceberg_cosine_similarity"]:
            if c not in g.columns:
                g[c] = 0.0
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
        g["_abs_ppm"] = g["ms1_error_ppm"].abs()
        e_top = g.sort_values(
            ["iceberg_entropy_similarity", "deepmet_sampling_frequency", "_abs_ppm"],
            ascending=[False, False, True], kind="stable").index[0]
        c_top = g.sort_values(
            ["iceberg_cosine_similarity", "deepmet_sampling_frequency", "_abs_ppm"],
            ascending=[False, False, True], kind="stable").index[0]
        e_v = g.at[e_top, "iceberg_entropy_similarity"]; c_v = g.at[e_top, "iceberg_cosine_similarity"]
        if e_top == c_top and ((e_v >= primary_min and c_v >= secondary_min) or (c_v >= primary_min and e_v >= secondary_min)):
            out.at[e_top] = True
    return out


def stage_pass(frame: pd.DataFrame, params) -> pd.Series:
    """Per-row pass flag for its own stage (A/B/C by fused threshold, D by the
    dual Top-1 rule).  Also writes ``level_score`` into *frame*."""
    if frame.empty:
        return pd.Series(False, index=frame.index)
    frame["level_score"] = frame.apply(lambda r: level_score(r, params), axis=1).astype(float)
    stage = frame["stage"].to_numpy()
    score = frame["level_score"].to_numpy(float)
    passed = np.zeros(len(frame), dtype=bool)
    enabled = enabled_levels(params)
    for name in ("A", "B", "C"):
        m = stage == name
        passed[m] = (score[m] >= _threshold(name, params)) & (name in enabled)
    md = stage == "D"
    if md.any() and "D" in enabled:
        passed[md] = clm_dual_top1(
            frame[md], params.clm_primary_min, params.clm_secondary_min
        ).to_numpy(dtype=bool)
    passed = pd.Series(passed, index=frame.index)
    return passed


def assign_levels(frame: pd.DataFrame, params, identity=None) -> pd.DataFrame:
    """Apply the A -> B -> C -> D waterfall to a candidate table.

    *frame* holds candidates of any stage for any spectra (a production run
    only computes later stages for spectra that are still unannotated; a
    calibration dump holds every stage for every spectrum -- the result is
    the same).  Returns the reported rows: for each spectrum only the
    passing candidates of the first passing level, one row per structure
    (best score kept), with ``confidence_level``, ``msi_level``,
    ``level_score`` and ``evidence_level`` set.  Spectra with no passing
    level are absent.

    *identity* maps a row to a structure key for the per-spectrum dedup
    (default: the ``smiles`` string).
    """
    if frame is None or frame.empty:
        return frame.iloc[0:0].copy() if frame is not None else pd.DataFrame()
    df = frame.copy()
    df["_pass"] = stage_pass(df, params)
    kept = []
    for _, group in df.groupby("spectrum_index", sort=False):
        for stage in LEVELS:
            hit = group[group["stage"].eq(stage) & group["_pass"]]
            if not hit.empty:
                kept.append(hit)
                break
    if not kept:
        return df.iloc[0:0].drop(columns=["_pass"])
    out = pd.concat(kept)
    out["confidence_level"] = out["stage"]
    out["msi_level"] = out["stage"].map(MSI_LEVEL)
    out["evidence_level"] = out["stage"].map(EVIDENCE_LEVEL)
    out["_rank_key"] = out.apply(rank_key, axis=1)
    out = out.sort_values(
        ["spectrum_index", "_rank_key"], ascending=[True, False], kind="stable"
    )
    key = identity or (lambda r: str(r.get("smiles", "") or ""))
    out["_struct"] = out.apply(key, axis=1)
    out = out.drop_duplicates(["spectrum_index", "_struct"], keep="first")
    return out.drop(columns=["_pass", "_struct"]).reset_index(drop=True)


def rank_key(row) -> tuple:
    """Order inside a spectrum: level, fused level score, then the stronger
    measured component, then explicit stereochemistry (MS/MS cannot resolve
    it, so it only breaks exact ties)."""
    stage = str(row.get("confidence_level", row.get("stage", "")) or "")
    third = 0.0
    if stage == "A":
        # pooled scores are shared by the stereo variants of one compound;
        # the variant's own direct spectral match orders them
        second = _num(row.get("a_hit_cosine", row.get("library_cosine")))
        third = _num(row.get("a_hit_dreams", row.get("DreaMS_similarity")))
    elif stage == "B":
        second = _num(row.get("library_cosine"))
    elif stage == "C":
        second = _num(row.get("mf_score"))
    else:
        second = _num(row.get("deepmet_sampling_frequency"))
    return (
        level_rank(stage),
        round(_num(row.get("level_score")), 6),
        round(second, 6),
        round(third, 6),
        stereo_specificity(row.get("smiles")),
    )


def select_topk_per_spectrum(
    df: pd.DataFrame,
    k: int,
    order_col: str = "spectrum_rank",
    ascending: bool = True,
) -> pd.DataFrame:
    """Return at most K ranked rows per spectrum; K=0 returns all rows."""
    if df is None or len(df) == 0:
        return df
    k = int(k)
    if k < 0:
        raise ValueError("Top-K must be zero (all) or a positive integer")
    if "spectrum_index" not in df.columns or order_col not in df.columns:
        ordered = df.sort_values(order_col, ascending=ascending) if order_col in df.columns else df.copy()
        return (ordered if k == 0 else ordered.head(k)).reset_index(drop=True)
    ordered = df.sort_values(
        ["spectrum_index", order_col], ascending=[True, ascending], kind="stable"
    )
    if k > 0:
        ordered = ordered.groupby("spectrum_index", sort=False).head(k)
    return ordered.reset_index(drop=True)
