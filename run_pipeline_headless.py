#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Only a QCoreApplication exists here; without Agg, matplotlib picks the Qt
# backend (PyQt5 is imported) and figure export aborts on "QWidget without QApplication".
os.environ.setdefault("MPLBACKEND", "Agg")

import hnswlib
import pandas as pd
from gensim.models import Word2Vec
from PyQt5 import QtCore

import annomass_app as appmod
from core.importing.load_from_files import load_from_files


def load_hnsw(path):
    dim = 300 if "spec2vec" in path.lower() else 200
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgf", default="Mgf_SXNI-P-30CE_fixed.mgf")
    ap.add_argument("--out", default="pipeline_headless_results.xlsx")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--detailed-export", action="store_true",
                    help="Export internal columns for benchmark/debug consumers instead of the compact public table.")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--skip-mf", action="store_true")
    ap.add_argument("--skip-figures", action="store_true",
                    help="Do not export ICEBERG fragment sheets and experimental/predicted mirror plots.")
    ap.add_argument("--per-spectrum-dedup", action="store_true",
                     help="Scope compound dedup to (spectrum_index, compound) instead of "
                          "pooling evidence across every spectrum in the batch. Use for "
                          "benchmark runs over many unrelated spectra; leave off for the "
                          "GUI's normal same-sample multi-scan use case.")
    ap.add_argument("--dreams-top-n", type=int, default=20,
                     help="DreaMS reference-library hits kept per query. Wider than the "
                          "20 default gives margin against batch-composition-dependent "
                          "embedding jitter dropping a near-tied reference row.")
    ap.add_argument("--dreams-batch-size", type=int, default=32,
                     help="DreaMS transformer minibatch size. 1 processes every query as "
                          "its own forward pass, eliminating batch-composition-dependent "
                          "embedding jitter at the source (slower than the 32 default).")
    ap.add_argument("--mf-adduct", default="[M+H]+",
                     help="Fallback Spectrum Predictor adduct when a spectrum does not carry "
                         "its own adduct metadata.")
    ap.add_argument("--iceberg-nce", type=float, default=30.0,
                    help="ICEBERG collision energy (default 30, matching DeepMet validation).")
    ap.add_argument("--iceberg-instrument", default="Unknown",
                    choices=["Unknown", "Orbitrap", "QTOF", "IT-FT"],
                    help="ICEBERG instrument category (default Unknown, matching DeepMet validation).")
    ap.add_argument("--iceberg-peak-min-rel", type=float, default=0.01,
                    help="Minimum peak intensity relative to each spectrum's base peak "
                         "for ICEBERG entropy/cosine scoring (default 0.01 = 1%%).")
    ap.add_argument("--iceberg-peak-top-k", type=int, default=80,
                    help="Maximum strongest peaks retained on each side for ICEBERG "
                         "entropy/cosine scoring (default 80; 0 keeps all above threshold).")
    ap.add_argument("--struct-db", default=None,
                     help="Override the structure candidate DB CSV path "
                          "(default: appmod.STRUCT_DB).")
    ap.add_argument("--ref-hdf5", default=None,
                     help="Override the DreaMS reference HDF5 path "
                          "(default: appmod.DEFAULT_REF_HDF5). For held-out "
                          "benchmark runs against a spectrum-blinded copy.")
    ap.add_argument("--index-pos", default=None,
                     help="Override the positive spec2vec hnswlib index path "
                          "(default: appmod.INDEX_POS). For validating a "
                          "merged/expanded DeepMASS reference library.")
    ap.add_argument("--ref-pos", default=None,
                     help="Override the positive DeepMASS reference pickle "
                          "path (default: appmod.REF_POS).")
    ap.add_argument("--level-a-weight", type=float, default=None,
                    help="Level A: weight of the DreaMS similarity (1 - weight goes to the "
                         "modified cosine to the matched library spectrum).")
    ap.add_argument("--level-a-threshold", type=float, default=None)
    ap.add_argument("--level-b-weight", type=float, default=None,
                    help="Level B: weight of the DeepMASS score.")
    ap.add_argument("--level-b-library-weight", type=float, default=None,
                    help="Level B: weight of the library-spectrum cosine; the DreaMS similarity "
                         "carried from level A gets 1 - level-b-weight - this.")
    ap.add_argument("--level-b-threshold", type=float, default=None)
    ap.add_argument("--level-c-weight", type=float, default=None,
                    help="Level C: weight of the carried level-B score (1 - weight goes to the "
                         "ICEBERG predicted-spectrum similarity).")
    ap.add_argument("--level-c-threshold", type=float, default=None)
    ap.add_argument("--ref-pos-exclude", default=None,
                    help="Benchmark blinding only: .npy of DeepMASS library positions to "
                         "ignore in the level-B library-spectrum lookup (written by "
                         "scripts/eval/build_blinded_refs_for_queries.py).")
    ap.add_argument("--evaluate-all-stages", action="store_true",
                    help="Calibration: compute every level for every spectrum (combine "
                         "with ANNOMASS_DUMP_STAGE_CANDIDATES=<csv>).")
    ap.add_argument("--modules", default="ABCD",
                    help="Modules to run, in waterfall order: A = DreaMS library retrieval, "
                         "B = DeepMASS candidates + experimental library spectra, C = DeepMASS "
                         "candidates + ICEBERG predicted spectra, D = DeepMet CLM. Examples: "
                         "--modules A (DreaMS only), --modules BC (DeepMASS only), --modules A,C. "
                         "Names also work: dreams, library, ice, clm, all. Default ABCD.")
    ap.add_argument("--min-fragment-peaks", type=int, default=3,
                    help="Exclude spectra with fewer real fragment peaks (>=1%% base, >2 Da from precursor); "
                         "0 disables (reproduces pre-2026-10-03 behaviour).")
    ap.add_argument("--benchmark-all-ms2", action="store_true",
                     help="Benchmark mode: forward every MS2 spectrum to DreaMS "
                          "even when DeepMASS produced no annotation.")
    ap.add_argument("--enable-deepmet-clm", dest="enable_deepmet_clm",
                    action="store_true", default=True,
                    help="Level D: spectra not annotated at A/B/C go to the DeepMet CLM "
                         "external-candidate route (default on).")
    ap.add_argument("--disable-deepmet-clm", dest="enable_deepmet_clm",
                    action="store_false",
                    help="Disable the DeepMet CLM route, e.g. for ablations.")
    ap.add_argument("--deepmet-clm-top-n", type=int, default=0,
                    help="CLM structures kept per mass window before ICEBERG; 0 keeps all "
                         "and is required for unbiased entropy/cosine Top-1 selection.")
    ap.add_argument("--enable-fiddle", dest="enable_fiddle",
                    action="store_true", default=False,
                    help="Optional: pre-filter library candidates with FIDDLE deep-learning "
                         "formula prediction (top-5). Off by default; formula screening via "
                         "declared metadata and exact-mass recall runs regardless.")
    ap.add_argument("--enable-abcort-rt", dest="enable_abcort_rt",
                    action="store_true", default=False,
                    help="Predict candidate retention times with ABCoRT and append "
                         "predicted_rt_min. RT is informational and does not affect ranking.")
    ap.add_argument("--abcort-checkpoint", default=None,
                    help="Optional ABCoRT checkpoint; defaults to the pretrained SMRT model.")
    ap.add_argument("--fiddle-instrument", default="Orbitrap",
                    choices=["Orbitrap", "Q-TOF"],
                    help="FIDDLE instrument model used with --enable-fiddle.")
    ap.add_argument("--disable-annosm-rerank", dest="enable_annosm_rerank",
                    action="store_false", default=True,
                    help="Disable the Step 6 AnnoSM fragment-evidence rerank/challenger "
                         "promotion, e.g. for ablations.")
    ap.add_argument("--annosm-max-level-gap", type=float, default=None,
                    help="AnnoSM promotion gate: a challenger is promoted only if its level score is within this "
                         "gap of the rank-1 (default: AnnoMassParams.annosm_max_level_gap = 0.02; 1.0 = ungated, for sensitivity analysis).")
    ap.add_argument("--annosm-exclude-external", dest="annosm_include_external",
                    action="store_false", default=True,
                    help="Let DeepMet CLM (category D) rows bypass the AnnoSM gated rerank "
                         "(default: they join it like library rows).")
    args = ap.parse_args()
    from core.staged_levels import parse_modules
    try:
        args.modules = parse_modules(args.modules)
    except ValueError as exc:
        ap.error(str(exc))
    if "D" in args.modules and args.modules != "ABCD" and not args.enable_deepmet_clm:
        print("[headless] note: module D selected but --disable-deepmet-clm given; D will not run")

    if args.skip_mf:
        appmod.mf_available = lambda: False

    qt_app = QtCore.QCoreApplication(sys.argv)

    print("[headless] loading spectra", flush=True)
    spectrums = load_from_files([args.mgf])
    for sp in spectrums:
        sp.set("only_flavonoids", False)
        sp.set("flav_classes", [])
        sp.set("allowed_elements", "")
        sp.set("max_candidates", 500)
        sp.set("use_fiddle", bool(args.enable_fiddle))
        sp.set("fiddle_instrument", args.fiddle_instrument)
        sp.set("fiddle_elements", "")

    print("[headless] loading resources", flush=True)
    pp = load_hnsw(args.index_pos or appmod.INDEX_POS)
    pn = load_hnsw(appmod.INDEX_NEG)
    deepmass_pos = Word2Vec.load(appmod.W2V_POS)
    deepmass_neg = Word2Vec.load(appmod.W2V_NEG)
    with open(args.ref_pos or appmod.REF_POS, "rb") as f:
        ref_pos = list(pickle.load(f))
    with open(appmod.REF_NEG, "rb") as f:
        ref_neg = list(pickle.load(f))
    db = pd.read_csv(args.struct_db or appmod.STRUCT_DB, low_memory=False)

    level_overrides = {
        k: v for k, v in {
            "level_a_weight": args.level_a_weight, "level_a_threshold": args.level_a_threshold,
            "level_b_weight": args.level_b_weight, "level_b_threshold": args.level_b_threshold,
            "level_b_library_weight": args.level_b_library_weight,
            "level_c_weight": args.level_c_weight, "level_c_threshold": args.level_c_threshold,
        }.items() if v is not None
    }
    worker = appmod.PipelineWorker(
        spectrums=spectrums,
        pp=pp,
        pn=pn,
        deepmass_pos=deepmass_pos,
        deepmass_neg=deepmass_neg,
        ref_positive=ref_pos,
        ref_negative=ref_neg,
        default_database=db,
        ref_hdf5=str(args.ref_hdf5 or appmod.DEFAULT_REF_HDF5),
        mz_tol_ppm=args.ppm,
        cho_only=True,
        mf_adduct=args.mf_adduct,
        mf_nce=args.iceberg_nce,
        mf_instrument=args.iceberg_instrument,
        iceberg_peak_min_rel_intensity=args.iceberg_peak_min_rel,
        iceberg_peak_top_k=args.iceberg_peak_top_k,
        enable_annosm_rerank=args.enable_annosm_rerank,
        **({} if args.annosm_max_level_gap is None else {"annosm_max_level_gap": args.annosm_max_level_gap}),
        annosm_include_external=args.annosm_include_external,
        top_n_per_spectrum=args.top_n,
        dedup_per_spectrum=args.per_spectrum_dedup,
        dreams_top_n=args.dreams_top_n,
        dreams_batch_size=args.dreams_batch_size,
        benchmark_all_ms2=args.benchmark_all_ms2,
        min_fragment_peaks=args.min_fragment_peaks,
        use_deepmet_clm=args.enable_deepmet_clm,
        deepmet_clm_top_n=args.deepmet_clm_top_n,
        enable_abcort_predict=args.enable_abcort_rt,
        abcort_checkpoint_path=args.abcort_checkpoint,
        ref_pos_exclude=args.ref_pos_exclude,
        evaluate_all_stages=args.evaluate_all_stages,
        enabled_levels=args.modules,
        **level_overrides,
    )

    result = {"payload": None, "error": None}

    def on_progress(msg):
        print("[pipeline]", msg, flush=True)

    def on_done(payload):
        result["payload"] = payload
        qt_app.quit()

    def on_failed(msg):
        result["error"] = msg
        qt_app.quit()

    worker.progress.connect(on_progress)
    worker.finished_with_result.connect(on_done)
    worker.failed.connect(on_failed)

    print("[headless] starting pipeline", flush=True)
    worker.start()
    qt_app.exec_()
    worker.wait()

    if result["error"]:
        raise RuntimeError(result["error"])

    payload = result["payload"]
    deepmet_df = pd.DataFrame()
    prediction_cache = {}
    if isinstance(payload, tuple):
        df = payload[0]
        if len(payload) >= 2 and payload[1] is not None:
            prediction_cache = payload[1]
        if len(payload) >= 3 and payload[2] is not None:
            deepmet_df = payload[2]
    else:
        df = payload
    if df is None:
        df = pd.DataFrame()
    from core.result_table_export import save_result_table
    out_lower = str(args.out).lower()
    if args.detailed_export:
        if out_lower.endswith(".csv"):
            df.to_csv(args.out, index=False, encoding="utf-8-sig")
        else:
            df.to_excel(args.out, index=False)
    else:
        save_result_table(df, args.out)
    print(f"[headless] saved {len(df)} rows to {args.out}", flush=True)
    out_path = os.path.abspath(args.out)
    stem, _ = os.path.splitext(out_path)
    # Confirmed external (category D) structures are part of the main workbook;
    # the third payload item is kept only for backward compatibility and is empty.
    if not deepmet_df.empty:
        deepmet_out = stem + "_generated_structure_inference.xlsx"
        save_result_table(deepmet_df, deepmet_out)
        print(
            f"[headless] saved {len(deepmet_df)} generated-structure inferences "
            f"to {deepmet_out}",
            flush=True,
        )
    if not args.skip_figures:
        from pathlib import Path
        from core.result_figure_exporter import export_result_figures
        parent = Path(out_path).parent
        figure_root = parent if parent.name == "final_results" else parent / "final_results"
        # Export figures for every final compound (library + DeepMet), not
        # only the external DeepMet sidecar when it is non-empty.
        figure_rows = df
        export_result_figures(
            figure_rows, spectrums, prediction_cache, figure_root,
            full_results=df, progress=lambda msg: print(msg, flush=True),
        )


if __name__ == "__main__":
    main()
