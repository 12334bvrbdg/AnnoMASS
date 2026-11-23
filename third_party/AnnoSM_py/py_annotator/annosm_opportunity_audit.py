#!/usr/bin/env python3
"""Audit which DeepMASS Top-1 errors are addressable by paper AnnoSM evidence."""

import argparse
from pathlib import Path

import pandas as pd

from .candidate_sm import extract_candidate_sm
from .mgf_parser import read_mgf
from .sm_annotator import SMAnnotator


def _ik14(value):
    return str(value)[:14]


def _find_column(df, token):
    return next((column for column in df.columns if token in column.lower()), None)


def _enrich_candidates(candidates, structure_paths):
    candidates = candidates.copy()
    candidates["_ik14"] = candidates["inchikey14"].map(_ik14)
    for structure_path in structure_paths:
        structures = pd.read_csv(structure_path, low_memory=False)
        ik_col = _find_column(structures, "inchi")
        smiles_col = _find_column(structures, "smile")
        if ik_col is None or smiles_col is None:
            continue
        structures["_ik14"] = structures[ik_col].map(_ik14)
        smiles_map = (
            structures.dropna(subset=[smiles_col])
            .drop_duplicates("_ik14")
            .set_index("_ik14")[smiles_col]
        )
        missing = candidates["smiles"].isna()
        candidates.loc[missing, "smiles"] = candidates.loc[missing, "_ik14"].map(
            smiles_map
        )
    return candidates


def _difference_type(top_sm, truth_sm):
    if top_sm.annotated_sm == truth_sm.annotated_sm:
        return "same_sm"
    if top_sm.raw_sm == truth_sm.raw_sm:
        return "a_ring_position"
    if top_sm.skeleton != truth_sm.skeleton:
        return "parent_skeleton"
    top_parts = top_sm.raw_sm.split("C", 1)
    truth_parts = truth_sm.raw_sm.split("C", 1)
    if top_parts[-1] != truth_parts[-1]:
        return "c_ring_substitution"
    return "a_b_ring_distribution"


def _candidate_evidence(annotator, peaks, candidate_sm, fragment_ppm):
    result = annotator.match_single_compound(
        compound_id="audit",
        precursor_mz=0,
        fragment_peaks=peaks,
        precursor_info={
            "OHsum": candidate_sm.oh_sum,
            "OCH3sum": candidate_sm.och3_sum,
            "Skeleton": candidate_sm.skeleton,
        },
        fragment_ppm=fragment_ppm,
    )
    if result.empty:
        return None, None, None
    hit = result[result["Substitution modes"] == candidate_sm.annotated_sm]
    if hit.empty:
        hit = result[result["Substitution modes"] == candidate_sm.raw_sm]
    if hit.empty:
        return 0.0, None, None
    row = hit.sort_values("Score", ascending=False).iloc[0]
    rank = int(result["Score"].rank(method="min", ascending=False).loc[row.name])
    expected_class = "ISO" if candidate_sm.skeleton == "IsoFlavones" else "FLA"
    observed_class = str(row.get("Class", "") or "")
    class_consistent = observed_class == expected_class
    return float(row["Score"]), rank, class_consistent


def audit(args):
    candidates = pd.read_csv(args.deepmass_csv, low_memory=False)
    candidates = _enrich_candidates(candidates, args.structure_db)
    truth = pd.read_csv(args.truth_csv, low_memory=False).reset_index(drop=True)
    truth["query_id"] = [f"test_{i}" for i in range(len(truth))]
    truth_map = dict(zip(truth["query_id"], truth["inchikey"].map(_ik14)))

    source_to_query = {
        str(row["source_id"]): row["query_id"] for _, row in truth.iterrows()
    }
    spectra = {}
    for spectrum in read_mgf(args.mgf):
        source_id = str(spectrum.get("meta", {}).get("SPECTRUMID", ""))
        query_id = source_to_query.get(source_id)
        if query_id:
            spectra[query_id] = spectrum["peaks"]

    annotator = SMAnnotator(args.cfil)
    valid_modes = set(annotator.cfil["Substitution modes"].dropna())
    rows = []

    for query_id, group in candidates.groupby("spectrum_title", sort=False):
        group = group.sort_values("rank").drop_duplicates("_ik14", keep="first")
        top = group.iloc[0]
        truth_ik = truth_map.get(query_id)
        if not truth_ik or top["_ik14"] == truth_ik:
            continue

        truth_rows = group[group["_ik14"] == truth_ik]
        truth_rank = int(truth_rows.iloc[0]["rank"]) if not truth_rows.empty else None
        truth_in_top_k = truth_rank is not None and truth_rank <= args.top_k
        truth_row = truth_rows.iloc[0] if truth_in_top_k else None

        top_sm = extract_candidate_sm(top.get("smiles"))
        truth_sm = extract_candidate_sm(truth_row.get("smiles")) if truth_row is not None else None
        paper_supported = (
            top_sm is not None
            and truth_sm is not None
            and top_sm.raw_sm in valid_modes
            and truth_sm.raw_sm in valid_modes
        )
        difference = (
            _difference_type(top_sm, truth_sm) if paper_supported else "not_addressable"
        )
        structurally_addressable = paper_supported and difference != "same_sm"

        top_score = top_rank = top_class_ok = None
        truth_score = truth_anno_rank = truth_class_ok = None
        peaks = spectra.get(query_id)
        if structurally_addressable and peaks is not None:
            top_score, top_rank, top_class_ok = _candidate_evidence(
                annotator, peaks, top_sm, args.fragment_ppm
            )
            truth_score, truth_anno_rank, truth_class_ok = _candidate_evidence(
                annotator, peaks, truth_sm, args.fragment_ppm
            )

        evidence_favors_truth = bool(
            structurally_addressable
            and truth_score is not None
            and top_score is not None
            and truth_score > top_score
            and (truth_class_ok is True or difference == "a_ring_position")
        )
        rows.append(
            {
                "query_id": query_id,
                "truth_ik14": truth_ik,
                "deepmass_top1_ik14": top["_ik14"],
                "truth_rank": truth_rank,
                "truth_in_top_k": truth_in_top_k,
                "top1_deepmass_score": top.get("deepmass_score"),
                "truth_deepmass_score": truth_row.get("deepmass_score") if truth_row is not None else None,
                "top1_sm": top_sm.annotated_sm if top_sm else None,
                "truth_sm": truth_sm.annotated_sm if truth_sm else None,
                "difference_type": difference,
                "paper_supported": paper_supported,
                "structurally_addressable": structurally_addressable,
                "top1_annosm_score": top_score,
                "truth_annosm_score": truth_score,
                "top1_annosm_rank": top_rank,
                "truth_annosm_rank": truth_anno_rank,
                "top1_class_consistent": top_class_ok,
                "truth_class_consistent": truth_class_ok,
                "evidence_favors_truth": evidence_favors_truth,
            }
        )

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output, index=False)

    print(f"DeepMASS Top-1 errors: {len(result)}")
    print(f"Truth in Top-{args.top_k}: {int(result['truth_in_top_k'].sum())}")
    print(f"Paper AnnoSM supported: {int(result['paper_supported'].sum())}")
    print(f"Structurally addressable: {int(result['structurally_addressable'].sum())}")
    print(f"Observed evidence favors truth: {int(result['evidence_favors_truth'].sum())}")
    print("\nAddressable differences:")
    print(result.loc[result["structurally_addressable"], "difference_type"].value_counts())
    print(f"\nSaved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", required=True)
    parser.add_argument("--deepmass-csv", required=True)
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--cfil", required=True, help="Original paper CFIL.xlsx")
    parser.add_argument("--structure-db", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fragment-ppm", type=float, default=50)
    parser.add_argument("-o", "--output", required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
