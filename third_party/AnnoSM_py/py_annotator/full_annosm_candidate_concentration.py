#!/usr/bin/env python3
"""Concentrate DeepMASS candidates with the complete paper AnnoSM workflow."""

import argparse
import hashlib
import re
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from .annosm_opportunity_audit import _enrich_candidates, _ik14
from .candidate_sm import extract_candidate_sm
from .mgf_parser import read_mgf
from .screener import CompoundScreener
from .sm_annotator import SMAnnotator


_UNCHARGER = rdMolStandardize.Uncharger()


def _normalize_formula(value):
    if value is None or pd.isna(value):
        return None
    return re.sub(r"\s+", "", str(value)).rstrip("+-")


def _candidate_formula(smiles):
    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        pass
    return _normalize_formula(rdMolDescriptors.CalcMolFormula(mol))


def _augment_tpil_with_candidates(tpil_path, candidates):
    """Append exact candidate precursor rows while preserving the paper screen."""
    path = Path(tpil_path)
    base = pd.read_excel(path) if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(path)
    base.columns = [str(column).strip() for column in base.columns]
    base["_candidate_augmented"] = False
    next_group = pd.to_numeric(base["Compound group"], errors="coerce").max()
    next_group = int(next_group) + 1 if pd.notna(next_group) else 1
    rows = []
    seen = set()
    for _, candidate in candidates.drop_duplicates("_ik14").iterrows():
        candidate_sm = extract_candidate_sm(candidate.get("smiles"))
        smiles = candidate.get("smiles")
        if candidate_sm is None or smiles is None or pd.isna(smiles):
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        try:
            mol = _UNCHARGER.uncharge(mol)
        except Exception:
            pass
        formula = _normalize_formula(rdMolDescriptors.CalcMolFormula(mol))
        exact_mass = float(rdMolDescriptors.CalcExactMolWt(mol))
        key = (formula, round(exact_mass, 6), candidate_sm.raw_sm)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "Compound group": next_group,
            "Class": "Aglycone",
            "OHsum": candidate_sm.oh_sum,
            "OCH3sum": candidate_sm.och3_sum,
            "Sugar and phenylpropionyl units": "-",
            "S&Psum": 0,
            "Formulas": formula,
            "Exact Mass": round(exact_mass, 4),
            "cal.m.z.H": round(exact_mass + 1.007276466, 4),
            "Skeleton": candidate_sm.skeleton,
            "_candidate_augmented": True,
        })
        next_group += 1
    if rows:
        added = pd.DataFrame(rows)
        for column in base.columns:
            if column not in added:
                added[column] = None
        for column in added.columns:
            if column not in base:
                base[column] = None
        base = pd.concat([base, added[base.columns]], ignore_index=True)
    handle = tempfile.NamedTemporaryFile(prefix="annosm_tpil_", suffix=".csv", delete=False)
    handle.close()
    base.to_csv(handle.name, index=False)
    return Path(handle.name), len(rows)


def _paper_candidate_evidence(
    annotator,
    peaks,
    screened_group,
    candidate_sm,
    formula,
    result_cache,
    fragment_ppm,
    score_column,
    specificity_weight,
    relax_tpil_composition,
):
    if candidate_sm is None:
        return "outside_paper_scope", 0.0, None, 0.0, 0.0, None, 0, 0.0, 0.0
    if formula is None:
        return "missing_formula", 0.0, None, 0.0, 0.0, None, 0, 0.0, 0.0
    if screened_group.empty:
        return "tpil_no_match", 0.0, None, 0.0, 0.0, None, 0, 0.0, 0.0

    formula_rows = screened_group[
        screened_group["_formula"] == formula
    ]
    compatible = formula_rows[
        (formula_rows["OHsum"] == candidate_sm.oh_sum)
        & (formula_rows["OCH3sum"] == candidate_sm.och3_sum)
    ]
    composition_relaxed = False
    if compatible.empty and relax_tpil_composition and not formula_rows.empty:
        compatible = formula_rows
        composition_relaxed = True
    if compatible.empty:
        return "tpil_incompatible", 0.0, None, 0.0, 0.0, None, 0, 0.0, 0.0
    key = (
        candidate_sm.skeleton,
        candidate_sm.oh_sum,
        candidate_sm.och3_sum,
    )
    if key not in result_cache:
        precursor = compatible.iloc[0].copy()
        precursor["Skeleton"] = candidate_sm.skeleton
        if composition_relaxed:
            precursor["OHsum"] = candidate_sm.oh_sum
            precursor["OCH3sum"] = candidate_sm.och3_sum
        result_cache[key] = annotator.match_single_compound(
            compound_id="candidate",
            precursor_mz=precursor.get("Exp.m/z", 0),
            fragment_peaks=peaks,
            precursor_info=precursor,
            fragment_ppm=fragment_ppm,
        )
    result_group = result_cache[key]
    if result_group.empty:
        return "cfil_no_match", 0.0, None, 0.0, 0.0, None, 0, 0.0, 0.0

    result_group = result_group.copy()
    if score_column == "HybridScore":
        result_group["HybridScore"] = (
            result_group["Score"]
            + specificity_weight * result_group["SpecificityScore"]
        )
    hit = result_group[result_group["Substitution modes"].isin(
        [candidate_sm.annotated_sm, candidate_sm.raw_sm]
    )]
    if hit.empty:
        return "cfil_incompatible", 0.0, None, 0.0, 0.0, None, 0, 0.0, 0.0

    row = hit.sort_values(score_column, ascending=False).iloc[0]
    mode_score = float(row[score_column])
    mode_count = len(result_group)
    mode_rank = int((result_group[score_column] > mode_score).sum() + 1)
    mode_percentile = (
        1.0
        if mode_count <= 1
        else 1.0 - (mode_rank - 1) / (mode_count - 1)
    )
    other_scores = result_group.drop(index=row.name)[score_column]
    margin_to_best_other = (
        mode_score - float(other_scores.max()) if not other_scores.empty else mode_score
    )
    observed_class = str(row.get("Class", "") or "")
    expected_class = (
        "ISO"
        if candidate_sm.skeleton in {"IsoFlavones", "Isoflavanone", "Isoflavan"}
        else "FLA"
    )
    if observed_class == expected_class:
        state = "fully_consistent"
    elif observed_class in {"", "nan"}:
        state = "sm_consistent_unclassified"
    else:
        state = "class_inconsistent"
    return (
        state,
        float(row[score_column]),
        observed_class,
        float(row["Score"]),
        float(row["SpecificityScore"]),
        mode_rank,
        mode_count,
        mode_percentile,
        margin_to_best_other,
    )


def _rank_group(group, args):
    group = group.sort_values("unique_rank").copy()
    if args.ranking_strategy == "gated":
        if args.anchor_k:
            raise ValueError("Gated ranking does not use pinned DeepMASS ranks")
        original_top1 = group.iloc[0]
        original_mode = re.sub(r"\([^)]*\)", "", str(original_top1["candidate_sm"]))
        challenger_modes = group["candidate_sm"].fillna("").map(
            lambda value: re.sub(r"\([^)]*\)", "", value)
        )
        covered_states = {"fully_consistent", "sm_consistent_unclassified"}
        if args.gated_state_policy == "both_full":
            original_is_covered = original_top1["annosm_state"] == "fully_consistent"
            challenger_is_covered = group["annosm_state"].eq("fully_consistent")
        else:
            original_is_covered = original_top1["annosm_state"] in covered_states
            challenger_is_covered = group["annosm_state"].isin(covered_states)
        challengers = group[
            (group["unique_rank"] > 1)
            & (group["formula"] == original_top1["formula"])
            & challenger_is_covered
            & challenger_modes.ne("")
            & challenger_modes.ne(original_mode)
            & (
                group["annosm_score"] - original_top1["annosm_score"]
                >= args.gated_min_annosm_margin
            )
            & (
                original_top1["deepmass_score"] - group["deepmass_score"]
                <= args.gated_max_deepmass_gap
            )
        ]
        if original_is_covered and not challengers.empty:
            promoted = challengers.sort_values(
                ["annosm_score", "deepmass_score", "unique_rank"],
                ascending=[False, False, True],
            ).head(1)
            group = pd.concat(
                [promoted, group.drop(index=promoted.index)], ignore_index=True
            )
        else:
            group = group.reset_index(drop=True)
        group["combined_score"] = group["deepmass_score"]
        group["concentrated_rank"] = range(1, len(group) + 1)
        group["selected"] = group["concentrated_rank"] <= args.output_k
        return group

    anchors = group.head(args.anchor_k).copy()
    remainder = group.iloc[args.anchor_k:].copy()
    incompatible = remainder["annosm_state"].isin({
        "tpil_no_match",
        "tpil_incompatible",
        "cfil_no_match",
        "cfil_incompatible",
        "class_inconsistent",
    })
    remainder["combined_score"] = (
        remainder["deepmass_score"]
        + args.evidence_weight * remainder["annosm_evidence"]
        - args.incompatible_penalty * incompatible.astype(float)
    )
    remainder = remainder.sort_values(
        ["combined_score", "unique_rank"], ascending=[False, True]
    )
    ranked = pd.concat([anchors, remainder], ignore_index=True)
    ranked["concentrated_rank"] = range(1, len(ranked) + 1)
    ranked["selected"] = ranked["concentrated_rank"] <= args.output_k
    return ranked


def concentrate(args):
    if not 0 <= args.anchor_k <= args.output_k <= args.input_k:
        raise ValueError("Require 0 <= anchor-k <= output-k <= input-k")

    candidates = pd.read_csv(args.deepmass_csv, low_memory=False)
    candidates = _enrich_candidates(candidates, args.structure_db)
    truth = pd.read_csv(args.truth_csv, low_memory=False).reset_index(drop=True)
    truth["query_id"] = [f"test_{i}" for i in range(len(truth))]
    truth_map = dict(zip(truth["query_id"], truth["inchikey"].map(_ik14)))
    source_map = dict(zip(truth["source_id"].astype(str), truth["query_id"]))

    spectra = []
    for spectrum in read_mgf(args.mgf):
        query_id = source_map.get(str(spectrum.get("meta", {}).get("SPECTRUMID", "")))
        if query_id:
            spectrum["title"] = query_id
            spectra.append(spectrum)

    tpil_path = Path(args.tpil)
    augmented_tpil_path = None
    augmented_tpil_rows = 0
    if args.augment_tpil_candidates:
        augmented_tpil_path, augmented_tpil_rows = _augment_tpil_with_candidates(
            tpil_path, candidates
        )
        tpil_path = augmented_tpil_path
    screener = CompoundScreener(tpil_path)
    annotator = SMAnnotator(args.cfil)
    screened_secondary = screener.screen_mgf(
        spectra,
        precursor_ppm=args.precursor_ppm,
        frag_ppm=args.precursor_ppm,
    )
    screened_primary = screener.screen_mgf(
        spectra,
        precursor_ppm=args.precursor_ppm,
        frag_ppm=args.precursor_ppm,
        use_fragment_refinement=False,
    )
    if augmented_tpil_path is not None:
        augmented_tpil_path.unlink(missing_ok=True)
    for screened_frame in (screened_secondary, screened_primary):
        if not screened_frame.empty:
            screened_frame["_formula"] = screened_frame["Formulas"].map(
                _normalize_formula
            )

    secondary_groups = {
        query_id: group
        for query_id, group in screened_secondary.groupby("Compound ID")
    } if not screened_secondary.empty else {}
    primary_groups = {
        query_id: group
        for query_id, group in screened_primary.groupby("Compound ID")
    } if not screened_primary.empty else {}
    paper_secondary_skeletons = {
        "Flavones",
        "Flavonols (C-3OH)",
        "Flavonols (C-3OCH3)",
        "IsoFlavones",
    }
    spectrum_map = {spectrum["title"]: spectrum["peaks"] for spectrum in spectra}

    rows = []
    for query_id, group in candidates.groupby("spectrum_title", sort=False):
        group = (
            group.sort_values("rank")
            .drop_duplicates("_ik14", keep="first")
            .head(args.input_k)
        )
        peaks = spectrum_map[query_id]
        result_cache = {}
        for unique_rank, (_, candidate) in enumerate(group.iterrows(), start=1):
            candidate_sm = extract_candidate_sm(candidate.get("smiles"))
            formula = _candidate_formula(candidate.get("smiles"))
            use_secondary = (
                candidate_sm is not None
                and candidate_sm.skeleton in paper_secondary_skeletons
            )
            screened_group = (
                secondary_groups if use_secondary else primary_groups
            ).get(query_id, pd.DataFrame())
            (
                state,
                score,
                observed_class,
                abundance_score,
                specificity_score,
                mode_rank,
                mode_count,
                mode_percentile,
                mode_margin,
            ) = _paper_candidate_evidence(
                    annotator,
                    peaks,
                    screened_group,
                    candidate_sm,
                    formula,
                    result_cache,
                    args.fragment_ppm,
                    args.annosm_score_column,
                    args.specificity_weight,
                    args.relax_tpil_composition,
                )
            if state == "fully_consistent":
                evidence = score
            elif state == "sm_consistent_unclassified":
                evidence = args.unclassified_weight * score
            else:
                evidence = 0.0
            rows.append({
                "query_id": query_id,
                "inchikey14": candidate["_ik14"],
                "compound_name": candidate.get("compound_name"),
                "smiles": candidate.get("smiles"),
                "formula": formula,
                "original_rank": int(candidate["rank"]),
                "unique_rank": unique_rank,
                "deepmass_score": float(candidate["deepmass_score"]),
                "candidate_sm": candidate_sm.annotated_sm if candidate_sm else None,
                "candidate_ohsum": candidate_sm.oh_sum if candidate_sm else None,
                "candidate_och3sum": candidate_sm.och3_sum if candidate_sm else None,
                "tpil_screening_path": "paper_secondary" if use_secondary else "primary",
                "annosm_state": state,
                "annosm_score": score,
                "annosm_abundance_score": abundance_score,
                "annosm_specificity_score": specificity_score,
                "annosm_mode_rank": mode_rank,
                "annosm_mode_count": mode_count,
                "annosm_mode_percentile": mode_percentile,
                "annosm_mode_margin": mode_margin,
                "annosm_class": observed_class,
                "annosm_evidence": evidence,
                "truth_ik14": truth_map.get(query_id),
            })

    scored = pd.DataFrame(rows)
    ranked = pd.concat(
        [_rank_group(group, args) for _, group in scored.groupby("query_id", sort=False)],
        ignore_index=True,
    )
    ranked["is_truth"] = ranked["inchikey14"] == ranked["truth_ik14"]
    evaluated = ranked["query_id"].nunique()

    top1_original = (
        ranked.sort_values(["query_id", "unique_rank"])
        .groupby("query_id", sort=False)
        .head(1)
        .set_index("query_id")
    )
    top1_concentrated = (
        ranked.sort_values(["query_id", "concentrated_rank"])
        .groupby("query_id", sort=False)
        .head(1)
        .set_index("query_id")
    )
    top1_changes = top1_original.add_prefix("original_").join(
        top1_concentrated.add_prefix("annosm_")
    )
    top1_changes = top1_changes[
        top1_changes["original_inchikey14"] != top1_changes["annosm_inchikey14"]
    ].reset_index()
    top1_changes["outcome"] = "wrong_to_wrong"
    top1_changes.loc[
        ~top1_changes["original_is_truth"] & top1_changes["annosm_is_truth"],
        "outcome",
    ] = "corrected"
    top1_changes.loc[
        top1_changes["original_is_truth"] & ~top1_changes["annosm_is_truth"],
        "outcome",
    ] = "harmed"

    top1_comparison = top1_original[["is_truth"]].rename(
        columns={"is_truth": "original_is_truth"}
    ).join(
        top1_concentrated[["is_truth"]].rename(
            columns={"is_truth": "annosm_is_truth"}
        )
    )
    top1_comparison["split"] = [
        "calibration"
        if int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16) % 10 < 4
        else "held_out"
        for query_id in top1_comparison.index
    ]
    validation_rows = []
    for split_name, split_group in [
        ("calibration", top1_comparison[top1_comparison["split"] == "calibration"]),
        ("held_out", top1_comparison[top1_comparison["split"] == "held_out"]),
        ("all", top1_comparison),
    ]:
        validation_rows.append({
            "split": split_name,
            "spectra": len(split_group),
            "original_top1": int(split_group["original_is_truth"].sum()),
            "annosm_top1": int(split_group["annosm_is_truth"].sum()),
            "delta": int(
                split_group["annosm_is_truth"].sum()
                - split_group["original_is_truth"].sum()
            ),
            "corrected": int(
                (~split_group["original_is_truth"] & split_group["annosm_is_truth"]).sum()
            ),
            "harmed": int(
                (split_group["original_is_truth"] & ~split_group["annosm_is_truth"]).sum()
            ),
            "ranking_strategy": args.ranking_strategy,
            "state_policy": args.gated_state_policy,
            "annosm_score_column": args.annosm_score_column,
            "specificity_weight": args.specificity_weight,
            "relax_tpil_composition": args.relax_tpil_composition,
            "min_annosm_margin": args.gated_min_annosm_margin,
            "max_deepmass_gap": args.gated_max_deepmass_gap,
        })
    gating_validation = pd.DataFrame(validation_rows)

    summary_rows = []
    for cutoff in sorted({1, 3, 5, 8, args.output_k, 12, 15, args.input_k}):
        if cutoff > args.input_k:
            continue
        original_hits = concentrated_hits = 0
        for _, group in ranked.groupby("query_id"):
            original_hits += bool(group.loc[group["unique_rank"] <= cutoff, "is_truth"].any())
            concentrated_hits += bool(
                group.loc[group["concentrated_rank"] <= cutoff, "is_truth"].any()
            )
        summary_rows.append({
            "cutoff": cutoff,
            "original_recall_count": original_hits,
            "original_recall": original_hits / evaluated,
            "concentrated_recall_count": concentrated_hits,
            "concentrated_recall": concentrated_hits / evaluated,
            "delta_count": concentrated_hits - original_hits,
        })
    summary = pd.DataFrame(summary_rows)

    cfil_states = {
        "fully_consistent",
        "sm_consistent_unclassified",
        "class_inconsistent",
        "cfil_incompatible",
    }
    cfil_evaluated = ranked[ranked["annosm_state"].isin(cfil_states)]
    coverage = pd.DataFrame([{
        "evaluated_spectra": evaluated,
        "tpil_secondary_spectra": screened_secondary["Compound ID"].nunique() if not screened_secondary.empty else 0,
        "tpil_secondary_rows": len(screened_secondary),
        "tpil_primary_spectra": screened_primary["Compound ID"].nunique() if not screened_primary.empty else 0,
        "tpil_primary_rows": len(screened_primary),
        "tpil_augmented_rows": augmented_tpil_rows,
        "candidate_cfil_evaluated_spectra": cfil_evaluated["query_id"].nunique(),
        "candidate_cfil_evaluated_rows": len(cfil_evaluated),
        "average_input_candidates": ranked.groupby("query_id").size().mean(),
        "average_selected_candidates": ranked[ranked["selected"]].groupby("query_id").size().mean(),
        "top1_changed_spectra": len(top1_changes),
        "top1_corrected_spectra": int((top1_changes["outcome"] == "corrected").sum()),
        "top1_harmed_spectra": int((top1_changes["outcome"] == "harmed").sum()),
    }])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        screened_secondary.drop(columns=["_formula"], errors="ignore").to_excel(
            writer, sheet_name="TPIL Secondary", index=False
        )
        screened_primary.drop(columns=["_formula"], errors="ignore").to_excel(
            writer, sheet_name="TPIL Primary", index=False
        )
        cfil_evaluated.to_excel(writer, sheet_name="Candidate CFIL Evidence", index=False)
        ranked[ranked["selected"]].to_excel(
            writer, sheet_name="Concentrated TopK", index=False
        )
        ranked.to_excel(writer, sheet_name="All Candidates", index=False)
        top1_changes.to_excel(writer, sheet_name="Top1 Changes", index=False)
        gating_validation.to_excel(writer, sheet_name="Gating Validation", index=False)

    print("Paper AnnoSM coverage:")
    print(coverage.to_string(index=False))
    print("\nCandidate recall:")
    print(summary.to_string(index=False))
    if args.anchor_k:
        print(f"\nOriginal Top-{args.anchor_k} is pinned and cannot change.")
    else:
        print("\nNo original DeepMASS ranks are pinned.")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", required=True)
    parser.add_argument("--deepmass-csv", required=True)
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--tpil", required=True)
    parser.add_argument("--cfil", required=True)
    parser.add_argument("--structure-db", action="append", default=[])
    parser.add_argument("--input-k", type=int, default=20)
    parser.add_argument("--output-k", type=int, default=10)
    parser.add_argument("--anchor-k", type=int, default=0)
    parser.add_argument(
        "--ranking-strategy", choices=["gated", "additive"], default="gated"
    )
    parser.add_argument(
        "--gated-state-policy",
        choices=["both_full", "both_covered"],
        default="both_covered",
    )
    parser.add_argument("--gated-min-annosm-margin", type=float, default=0.01)
    parser.add_argument("--gated-max-deepmass-gap", type=float, default=0.25)
    parser.add_argument(
        "--annosm-score-column",
        choices=["Score", "SpecificityScore", "HybridScore"],
        default="HybridScore",
    )
    parser.add_argument("--specificity-weight", type=float, default=0.1)
    parser.add_argument("--relax-tpil-composition", action="store_true")
    parser.add_argument("--augment-tpil-candidates", action="store_true")
    parser.add_argument("--precursor-ppm", type=float, default=10)
    parser.add_argument("--fragment-ppm", type=float, default=50)
    parser.add_argument("--unclassified-weight", type=float, default=0.5)
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--incompatible-penalty", type=float, default=0.0)
    parser.add_argument("-o", "--output", required=True)
    concentrate(parser.parse_args())


if __name__ == "__main__":
    main()
