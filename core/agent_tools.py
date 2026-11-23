"""Tool registry + the pure tool-calling loop for the Flavonoid Agent.

RANKING BOUNDARY: none of these tools ever write to `_rank_key`,
`level_score`, `confidence_level`, or `main_window._pipeline_last_df`/
`_pipeline_last_df_full`'s contents or ordering, or call
`core.staged_levels.assign_levels` with modified inputs. Annotation levels and
ranking come only from the staged DreaMS -> DeepMASS -> CLM waterfall (see
core/staged_levels.py) -- this module is strictly read/dispatch-only.
`run_pipeline` (implemented in annomass_app.py, not here, since it needs
main-thread GUI state) takes zero parameters for the same reason: the agent
must never be able to change a ranking-relevant setting via chat.

Two "bridge" tools (`load_spectra_file`, `run_pipeline`) are declared here
only as schemas -- their actual implementation lives in annomass_app.py's
AgentWorker, because they must run via the cross-thread main-window bridge
(see AgentWorker._tool_run_pipeline / _tool_load_spectra_file). Every other
tool is safe to call directly from AgentWorker's own background thread.
"""

import json
import os
import py_compile
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core import abcort_bridge
from core import agent_memory
from core import agent_skills
from core import annosm_bridge
from core import flavonoid_reference_db
from core import massbank_lookup
from core import massql
from core import mona_lookup
from core import pubchem_lookup
from core import pubmed_search
from core import structure_db_query

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAX_CODE_WRITE_CHARS = 500_000
_CODE_EDIT_ALLOWED_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".css", ".html", ".js", ".ts", ".tsx", ".jsx", ".ui", ".qss", ".sh",
    ".csv", ".tsv", ".log",
}
_CODE_EDIT_DENY_PARTS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "model", "data",
}


def _file_stat_payload(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": int(st.st_size),
        "absolute_path": str(path.resolve()),
        "project_path": _project_relpath(path),
    }

# Handed to the LLM verbatim so it explains levels and ranking from the real
# rules instead of inventing its own description (see core/staged_levels.py).
CASCADE_EXPLANATION_TEXT = (
    "AnnoMASS annotates positive-ion spectra with a staged waterfall; each spectrum stops at "
    "the first level that yields a passing candidate and only that level's candidates are "
    "reported. Level A (MSI Level 2): DreaMS library retrieval; score = 0.8 x DreaMS similarity "
    "+ 0.2 x modified cosine between the query and the matched library spectrum, passing at >= 0.70. "
    "Hits are pooled per compound (planar structure, InChIKey first block): a compound takes its best "
    "DreaMS similarity and best library cosine over all its library spectra, so several spectra of one "
    "compound measured at different energies or on different instruments add up to one score; stereoisomers "
    "of that compound stay separate rows ordered by their own library cosine. Level B (MSI Level 2): when no A candidate passes, the "
    "DeepMASS structure candidates and the DreaMS hits that failed A form one pool; candidates "
    "with library evidence (a DreaMS hit or an experimental DeepMASS library spectrum) are "
    "scored as 0.6 x DeepMASS score + 0.1 x best direct library-spectrum cosine + 0.3 x the DreaMS "
    "similarity carried down from A, passing at >= 0.97. Level C (MSI Level 3): when no B candidate passes, "
    "every candidate (including those that failed B) carries its B score down and is scored as "
    "0.3 x that B score + 0.7 x the ICEBERG predicted-vs-measured spectrum similarity, passing at >= 0.12. "
    "Level D (MSI Level 4): DeepMet CLM structures absent from "
    "the structure database, kept when ICEBERG entropy and cosine similarity are both >= 0.5 "
    "and pick the same Top-1 structure. Spectra that pass no level are not annotated. The precursor "
    "mass tolerance is 10 ppm plus the rounding error of the decimals recorded in the file. Weights "
    "and thresholds are fixed values: the B/C weights and thresholds and the D rule come from a benchmark-disjoint "
    "calibration set; the level-A weight (0.8) and threshold (0.70) and the AnnoSM gap below were chosen "
    "with the main-benchmark Top-1 in view (no benchmark Top-1 was allowed to drop), so say so if asked how they "
    "were obtained. Inside a "
    "spectrum, candidates are ordered by their level score; AnnoSM may swap same-formula "
    "isomers of the same level only when its challenger's level score is within 0.02 of the rank-1 and "
    "its gate conditions hold. The user can run only some of the levels (A, B, C, D checkboxes "
    "in Settings; --modules in the terminal): a skipped level is not computed and never passes, "
    "and without A the B score is re-weighted over the DeepMASS score and the library cosine; the "
    "result column modules_run records which levels ran, so check it before explaining why a spectrum "
    "is unannotated. MSI Level 1 needs an in-house "
    "authentic standard and is never assigned automatically."
)

DEEPMET_ROUTE_EXPLANATION_TEXT = (
    "The DeepMet CLM route is level D, the last step of the waterfall: it only runs for "
    "spectra that no library level (A, B, C) could annotate. The CLM is an unconditional "
    "generator fine-tuned on flavonoids; the spectrum is not a model input, so the route "
    "queries a pre-generated, strictly filtered theoretical pool by precursor mass (10 ppm) "
    "and removes structures already in the structure database. Every remaining structure is "
    "scored with ICEBERG, and one structure per spectrum is kept as level D only when it is "
    "Top-1 by both entropy and cosine similarity and both scores are at least 0.5. D "
    "structures carry no stereochemistry and have no reference-standard or NMR confirmation "
    "(MSI Level 4)."
)

SYSTEM_PROMPT = (
    "You are the Flavonoid Agent, a chat assistant embedded in a flavonoid "
    "mass-spectrometry compound-identification desktop app (AnnoMASS). You "
    "have tools to explain already-computed AnnoMASS results, export result "
    "files, run the AnnoMASS algorithm, load a spectra file, and query "
    "the project's flavonoid reference/structure databases. When explicitly "
    "asked, you can also inspect and edit project code with guarded "
    "Codex-lite tools. You can also run "
    "MassQL queries for spectrum interpretation: use query_massql for the "
    "currently loaded spectra, query_massql_file for a local MGF/mzML/mzXML "
    "library file, query_massql_gnps for GNPS2/online MassQL parsing and "
    "remote workflow links, query_massql_libraries when the user wants broad "
    "online spectral-library/public-data routing including MoNA links, and "
    "query_massql_remote_file only when the user explicitly provides a direct "
    "downloadable MS data URL. Use query_mona_online for live MoNA spectral-library "
    "lookup by compound name, InChIKey, SPLASH, precursor m/z, or raw MoNA RSQL; "
    "if MoNA requires authorization, report the returned browser/download links and "
    "suggest downloading a MoNA export for local MassQL querying. "
    "Current AnnoMASS workflow (positive ion only; negative-ion spectra are skipped): input "
    "quality control keeps fragment peaks >=1% of the base peak and >2 Da from the precursor, "
    "and spectra with fewer than 3 such peaks are not analysed by default. Each spectrum then "
    "walks a fixed waterfall and stops at the first level that yields a passing candidate: "
    "A = DreaMS library match (DreaMS similarity + spectral cosine to the matched library "
    "spectrum, MSI Level 2); B = DeepMASS candidate or DreaMS hit that failed A, with library "
    "evidence (DeepMASS score + library cosine + carried DreaMS score, MSI Level 2); C = any "
    "candidate, including those that failed B (carried B score + ICEBERG similarity, MSI Level 3); D = DeepMet "
    "CLM library-external structure (ICEBERG entropy and cosine dual Top-1, both >= 0.5, MSI "
    "Level 4). Pooling, weights and the user's module selection (modules_run column) are in the cascade explanation "
    "text returned by explain_pipeline_results. A spectrum that passes no level is not annotated and has no row. Only the "
    "candidates of the spectrum's own level are reported; there are no E-H labels any more. "
    "Each level score is a fixed weighted sum with a fixed threshold (calibrated once, not "
    "user-tunable via chat). Do not describe DeepMet as running in parallel with the library "
    "route, and do not call it MetGenX. The GUI Save Results button and headless pipeline "
    "export experimental-vs-ICEBERG mirror spectra and matched ICEBERG fragment-structure "
    "sheets for rows that have a predicted spectrum (C and D); these are fragment structures, "
    "not a fragmentation tree. The pipeline output is not capped; Top-K (GUI default 3 per "
    "spectrum) is a post-hoc display filter. "
    "If the user sends a MassQL-looking command (e.g. starts with QUERY or "
    "WHERE MS2PROD/MS2NL/MS1MZ/RT), run the appropriate MassQL tool rather "
    "than merely explaining the syntax. query_pubchem, "
    "search_literature, and query_massbank_online are LIVE internet lookups "
    "(PubChem/PubMed/massbank.eu) -- try the local query_structure_database/"
    "query_reference_spectra tools first since they're instant and offline; "
    "use the internet tools as a fallback or to supplement with broader "
    "coverage, and say so when you use them. query_massbank_online returns "
    "REAL external MS/MS records (not just structural metadata like "
    "PubChem, and not literature prose like PubMed) -- still treat a match "
    "there as 'another lab has a reference spectrum for this compound', NOT "
    "as automatic confirmation that it matches the user's current unknown "
    "spectrum; this tool doesn't compare peaks against the loaded data, "
    "query_reference_spectra/explain_pipeline_results do that.\n\n"
    "Literature vs. spectrum evidence -- keep them visibly separate, never "
    "blend them into one claim: PubChem gives structural metadata, not "
    "spectral or retention-time data about YOUR sample. A PubMed abstract "
    "describes a DIFFERENT study/instrument/sample -- if an abstract states "
    "a general fragmentation rule (e.g. 'RDA fragment at m/z X indicates "
    "3-O substitution'), that's literature-based context you can offer, but "
    "it is NOT verified against the current spectrum unless a tool "
    "(explain_pipeline_results/compare_isomers/AnnoSM) actually found that "
    "fragment in THIS data. Label literature-derived statements explicitly "
    "(e.g. 'literature suggests...', 'according to a paper on...') and never "
    "present them with the same certainty as a tool-confirmed evidence_level "
    "or fragment match -- conflating the two would let an unverified claim "
    "quietly become 'evidence' for this specific identification.\n\n"
    "Export/report behavior: when the user asks to export/save/generate a result "
    "file, call export_pipeline_report directly. When the user asks for a Word "
    "document, method section, results section, manuscript-style report, or "
    "method/results write-up, call write_pipeline_word_report. Do not only "
    "describe how to do it. Use CSV unless the user asks for Excel/XLSX, "
    "Markdown/report text, or Word/docx. Only when the user explicitly asks "
    "for a filtered subset (for example category A, a spectrum-rank range, one "
    "class, or one spectrum), pass export_pipeline_report.filters instead of "
    "telling the user to filter manually or asking for Code mode. When the user explicitly asks "
    "to draw/plot/visualize AnnoMASS results, call plot_pipeline_results and "
    "return/open the generated PDF/PNG/SVG rather than only describing a "
    "plotting recipe; do not make figures proactively without a plotting "
    "request. Generated AnnoMASS reports and figures are saved under the "
    "project-local reports/agent_reports directory, not under ~/.annomass.\n\n"
    "Conversational preview saving: when the user asks to save the spectrum, structure, image, "
    "or file currently shown in the Agent preview to a particular folder or under a particular "
    "name, call save_current_preview with that exact destination path. Do not merely tell the user "
    "to click Save As. Do not overwrite an existing file unless the user explicitly requested it.\n\n"
    "Visible process: the UI shows your plan/tool-call timeline and tool inputs/"
    "outputs. Do not reveal hidden chain-of-thought; give concise rationale, "
    "evidence, and action logs instead.\n\n"
    "Skills: installed Agent skills are Markdown instructions loaded into this "
    "system prompt. Use list_agent_skills to inspect them. Use install_agent_skill "
    "only when the user explicitly asks to install/update a reusable skill or "
    "provides skill content/path.\n\n"
    "Code-editing behavior: you have Codex-lite project tools. Use them only "
    "when the user explicitly asks you to inspect, modify, debug, or validate "
    "the GUI/project code. Before editing, read/search the relevant files. "
    "Tools that modify files or run validation commands require an explicit "
    "user permission dialog in the GUI; if permission is denied, explain that "
    "the action was not run and continue with what can be done safely. "
    "For multi-file or surgical edits, prefer apply_project_patch over whole-"
    "file rewrites. After writing or applying a patch, call get_project_diff "
    "and run review_project_code plus run_project_tests or another allowed "
    "validation command when applicable. When the user asks for code self-"
    "check, code review, 自查, or review, call review_project_code and report "
    "findings by severity with file/line references. Do not touch ranking "
    "logic casually; if a requested code change could alter candidate ranking "
    "or scientific evidence interpretation, explain the risk and keep the "
    "change minimal and transparent. Never claim you ran a command unless "
    "run_project_command or run_project_tests returned the result. After any "
    "file-generating tool call, only say the file was created or exists when "
    "the tool result includes verified=true (or the dedicated open_file tool "
    "successfully opened an already-existing file); otherwise report the "
    "tool's error/uncertainty instead of inferring success. "
    "After any "
    "code edit, your final reply must be Codex-style and concise: say what "
    "changed, list the files touched, say what validation/review commands "
    "were run and whether they passed, and mention any remaining risk or "
    "anything you could not verify. Do not paste large diffs unless asked.\n\n"
    "Hard rule: annotation levels and candidate ranking come from a fixed staged "
    "waterfall (fixed weights and thresholds) you "
    "have no ability to change and must never claim to change. Only explain "
    "or report on results using data returned by your tools -- never invent "
    "scores, evidence levels, or reference-library facts that a tool did not "
    "return. If a tool returns an error (e.g. no spectra loaded yet), relay "
    "that plainly to the user instead of guessing what might fix it silently.\n\n"
    "Memory: call remember_fact when you learn something that would help a "
    "FUTURE conversation -- a stated user preference, a recurring project "
    "fact (e.g. 'this MGF batch only has positive-mode spectra'), something "
    "you'd otherwise have to be told again. Do NOT call it for one-off task "
    "details, or anything already retrievable from your other tools -- that's "
    "not memory, that's just re-running a query.\n\n"
    "Evidence categories: explain_pipeline_results returns each candidate's "
    "evidence pre-grouped into spectrum_evidence (annotation level A-D, MSI level, level "
    "score, DreaMS similarity, library-spectrum cosine, DeepMASS score), "
    "fragment_evidence (ICEBERG predicted spectrum/AnnoSM), and orthogonal_evidence "
    "(predicted retention time -- always relay its system-dependent caveat "
    "when you cite it). When explaining a candidate in any depth, organize "
    "your answer along these same 3 categories rather than one flat "
    "paragraph. "
    "Note that 'conflicting evidence' between two candidates is NOT one of "
    "these 3 (no single candidate's evidence 'conflicts' with itself) -- "
    "use compare_isomers for that.\n\n"
    "Evidence labels are not accuracy estimates: never quote a per-label Top-1 accuracy "
    "unless a tool returned it. Be explicit about what each candidate's evidence is -- "
    "a level-C candidate has no experimental reference spectrum (only a DeepMASS structure "
    "prior plus an ICEBERG predicted spectrum, MSI Level 3), and a level-D candidate is a "
    "generated structure with no reference spectrum, standard or NMR confirmation (MSI "
    "Level 4). Describe both more cautiously than an A or B result (MSI Level 2) and say "
    "what evidence is missing.\n\n"
    "AnnoSM interpretation guard: NEVER say AnnoSM 'ranked down', 'demoted', "
    "or 'judged down' a candidate merely because annosm_state is "
    "cfil_incompatible/tpil_incompatible or annosm_score is 0. Those fields "
    "are diagnostic fragment-compatibility annotations. AnnoSM changed the "
    "ranking order only when annosm_promoted is true for a candidate; if "
    "annosm_promoted is false, explain ranking from selection_trace (level "
    "score) instead.\n\n"
    "Isomer disambiguation: when two ranked candidates share a molecular "
    "formula (classic positional/glycosidic isomers), compare_isomers "
    "returns diagnostic_fragments -- the specific CFIL characteristic-"
    "fragment-ion m/z values and their matched intensity in the query "
    "spectrum that support each side's substitution mode (this is AnnoSM's "
    "real ion-abundance evidence, broken down from its aggregate score, not "
    "an invented rule). Cite those m/z/intensity values directly when "
    "explaining why one substitution mode is better supported than the "
    "other. For a stronger, independent check, query_massbank_online with "
    "each candidate's own compound name/InChIKey to see whether a real "
    "external reference spectrum for that EXACT isomer exists; a genuine "
    "reference spectrum is stronger evidence than any derived rule, but "
    "still relay it as 'another lab's reference spectrum for this specific "
    "compound', not as a re-ranking of the pipeline's own results."
)

# How proactively the agent reaches for extra verification (e.g. the
# MassBank cross-check above) vs. answering directly -- a global,
# provider-independent behavior lever the user sets in Agent Settings (see
# core/agent_config.py's THINKING_EFFORT_LEVELS). Appended to SYSTEM_PROMPT
# at call time (annomass_app.py's AgentWorker.run()), not baked into the
# constant, so changing the setting takes effect on the next message
# without restarting. Deliberately a PROMPT instruction, not a provider-
# native "reasoning"/"thinking" API parameter: most providers configurable
# here (DeepSeek/Qwen/Doubao/custom) have no such parameter, and a native
# extended-thinking feature would itself spend MORE tokens -- the opposite
# of what "low" is for. "low" saves tokens for real (fewer tool calls,
# terser prose); "high" spends more only when it buys real correctness.
EFFORT_INSTRUCTIONS = {
    "low": (
        "\n\n[Thinking effort: LOW -- minimize token spend] Answer as "
        "directly as possible. Only call a tool when you genuinely don't "
        "already have the data needed to answer. Skip extra cross-checks "
        "(e.g. the MassBank isomer lookup above) unless the user explicitly "
        "asks for stronger verification. Keep prose short and to the point."
    ),
    "medium": (
        "\n\n[Thinking effort: MEDIUM -- balanced, default] Use your "
        "judgment: call the tools needed to answer correctly, but don't "
        "chase extra verification the user didn't ask for once the answer "
        "is already well-supported by what you have."
    ),
    "high": (
        "\n\n[Thinking effort: HIGH -- prioritize correctness over token "
        "cost] For ambiguous cases -- especially same-formula isomer "
        "candidates -- proactively cross-check with multiple independent "
        "sources before answering: call compare_isomers for its "
        "diagnostic_fragments evidence, then query_massbank_online for each "
        "candidate to check for a real reference spectrum of that exact "
        "compound. Don't stop at the first plausible answer when stronger "
        "evidence is one more tool call away."
    ),
}

_PIPELINE_RESULT_COLUMNS = [
    "spectrum_title", "spectrum_index", "spectrum_rank",
    "compound_name", "rank", "confidence_level", "msi_level", "level_score",
    "evidence_level", "candidate_source", "formula", "smiles",
    "support_spectra_count", "spectrum_ids", "support_precursor_mz",
    "DreaMS_similarity", "library_cosine", "deepmass_score",
    "spectrum_predictor_score", "iceberg_entropy_similarity", "iceberg_cosine_similarity",
    "annosm_state", "annosm_score", "annosm_promoted", "modules_run",
    "predicted_rt_min", "flavonoid_class", "query_adduct",
    "ms1_error_ppm", "ms2_peaks", "source",
]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    thread_mode: str  # "direct" | "bridge"
    fn: Optional[Callable] = None  # None for "bridge" tools (implemented in annomass_app.py)


def _safe_records(df: pd.DataFrame, columns, limit=None) -> List[Dict[str, Any]]:
    cols = [c for c in columns if c in df.columns]
    out = df[cols].copy()
    if limit is not None:
        out = out.head(limit)
    # NaN -> None so json.dumps doesn't choke, and results read cleanly
    return out.where(pd.notna(out), None).to_dict(orient="records")


def _apply_result_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Small deterministic filter helper for Agent export/report tools."""
    if not filters:
        return df
    out = df
    for key, raw_value in (filters or {}).items():
        if key not in out.columns:
            continue
        if raw_value is None or raw_value == "":
            continue
        series = out[key]
        if isinstance(raw_value, (list, tuple, set)):
            wanted = {str(v).casefold() for v in raw_value}
            out = out[series.astype(str).str.casefold().isin(wanted)]
        elif isinstance(raw_value, dict):
            if "eq" in raw_value:
                wanted = str(raw_value["eq"]).casefold()
                out = out[series.astype(str).str.casefold() == wanted]
            if "contains" in raw_value:
                needle = str(raw_value["contains"]).casefold()
                out = out[series.astype(str).str.casefold().str.contains(needle, na=False, regex=False)]
            if "min" in raw_value:
                out = out[pd.to_numeric(series, errors="coerce") >= float(raw_value["min"])]
            if "max" in raw_value:
                out = out[pd.to_numeric(series, errors="coerce") <= float(raw_value["max"])]
        else:
            wanted = str(raw_value).casefold()
            out = out[series.astype(str).str.casefold() == wanted]
    return out


# ---------------------------------------------------------------------------
# Direct tools (safe to call from AgentWorker's own background thread)
# ---------------------------------------------------------------------------

def get_pipeline_status(main_window) -> Dict[str, Any]:
    spectrums = getattr(main_window, "spectrums", None)
    spectra_loaded = spectrums is not None and len(spectrums) > 0
    resource_attrs = ("pp", "pn", "deepmass_positive", "deepmass_negative",
                       "reference_positive", "reference_negative")
    resources_ready = all(getattr(main_window, attr, None) is not None for attr in resource_attrs)
    last_df = getattr(main_window, "_pipeline_last_df", None)
    pipeline_has_run = last_df is not None and len(last_df) > 0
    deepmet_df = getattr(main_window, "_pipeline_last_denovo_df", None)
    return {
        "spectra_loaded": spectra_loaded,
        "n_spectra": int(len(main_window.spectrums)) if spectra_loaded else 0,
        "resources_ready": resources_ready,
        "pipeline_has_run": pipeline_has_run,
        "n_candidates": int(len(last_df)) if pipeline_has_run else 0,
        "deepmet_clm_enabled": bool(
            getattr(getattr(main_window, "_pl_deepmet_clm_chk", None), "isChecked", lambda: False)()
        ),
        "n_final_deepmet_clm_structures": int(len(deepmet_df)) if deepmet_df is not None else 0,
        "n_spectra_with_results": (
            int(last_df["spectrum_index"].nunique())
            if pipeline_has_run and "spectrum_index" in last_df.columns else 0
        ),
    }


def check_spectrum_quality(main_window, min_peaks: int = 3) -> Dict[str, Any]:
    """Rule-based triage of loaded spectra BEFORE running AnnoMASS --
    flags spectra likely too sparse/malformed to yield a confident
    identification (too few MS2 peaks, missing precursor m/z), so the user
    can filter or fix them before spending a full run (and tokens) on data
    that was never going to work. Only simple, objective checks -- never a
    learned/composite quality score, same principle as ranking staying a
    transparent rule cascade."""
    spectrums = getattr(main_window, "spectrums", None)
    if spectrums is None or len(spectrums) == 0:
        return {"error": "No spectra loaded yet."}

    flagged = []
    n_ok = 0
    for i in range(len(spectrums)):
        row = spectrums.iloc[i]
        title = row.get("title", f"spectrum_{i}")
        sp = row.get("spectrum")
        issues = []
        n_peaks = 0
        precursor_mz = None
        if sp is None:
            issues.append("no spectrum object")
        else:
            try:
                n_peaks = len(sp.peaks.mz)
            except Exception:
                issues.append("could not read MS2 peaks")
            if n_peaks < min_peaks:
                issues.append(f"only {n_peaks} MS2 peak(s) (< {min_peaks})")
            try:
                precursor_mz = sp.get("precursor_mz")
            except Exception:
                precursor_mz = None
            if not precursor_mz:
                issues.append("missing precursor m/z")
        if issues:
            flagged.append({
                "spectrum_index": i,
                "title": str(title),
                "n_peaks": n_peaks,
                "precursor_mz": precursor_mz,
                "issues": issues,
            })
        else:
            n_ok += 1

    return {
        "n_total": int(len(spectrums)),
        "n_ok": n_ok,
        "n_flagged": len(flagged),
        "flagged": flagged,
        "note": (
            "These are simple objective checks (peak count, missing "
            "precursor m/z), not a learned quality score -- a flagged "
            "spectrum may still be worth running, it's just less likely to "
            "get a confident identification."
        ),
    }


def summarize_pipeline_results(main_window) -> Dict[str, Any]:
    """Aggregate statistics across the WHOLE current AnnoMASS run (not one
    candidate/spectrum) -- evidence_level / flavonoid_class / AnnoSM-state
    distributions, and how many spectra have a same-formula Top-1/Top-2 pair
    (isomer-ambiguous cases worth a compare_isomers follow-up). For when the
    user wants an overview of a batch run rather than drilling into one
    spectrum. Read-only, over the currently displayed Top-K-filtered table
    (main_window._pipeline_last_df) -- does not touch ranking."""
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}

    n_spectra = int(df["spectrum_index"].nunique()) if "spectrum_index" in df.columns else None

    def _counts(col):
        if col not in df.columns:
            return {}
        return df[col].fillna("(none)").astype(str).value_counts().to_dict()

    n_isomer_ambiguous = 0
    if {"spectrum_index", "rank", "formula"} <= set(df.columns):
        for _, group in df.groupby("spectrum_index"):
            top2 = group[group["rank"].isin([1, 2])].sort_values("rank")
            if len(top2) == 2 and top2.iloc[0]["formula"] and top2.iloc[0]["formula"] == top2.iloc[1]["formula"]:
                n_isomer_ambiguous += 1

    return {
        "n_spectra_with_results": n_spectra,
        "n_candidate_rows": int(len(df)),
        "evidence_level_distribution": _counts("evidence_level"),
        "flavonoid_class_distribution": _counts("flavonoid_class"),
        "annosm_state_distribution": _counts("annosm_state"),
        "n_spectra_with_isomer_ambiguous_top2": n_isomer_ambiguous,
        "note": "Counts are over the currently displayed Top-K-filtered results.",
    }


def filter_pipeline_results(
    main_window,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Preview filtered AnnoMASS rows without writing a file."""
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}
    source_rows = int(len(df))
    filtered = _apply_result_filters(df, filters)
    limit = max(1, min(int(limit or 20), 100))
    return {
        "source_rows": source_rows,
        "n_rows": int(len(filtered)),
        "filters": filters or {},
        "rows": _safe_records(filtered, _PIPELINE_RESULT_COLUMNS, limit=limit),
        "note": "Use export_pipeline_report with the same filters to save this subset.",
    }


_REPORTS_DIR = _PROJECT_ROOT / "reports" / "agent_reports"


def _writable_reports_dir() -> Path:
    for path in (_REPORTS_DIR, _PROJECT_ROOT / "temp" / "agent_reports", Path("/tmp") / "deepmass2_agent_reports"):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except Exception:
            continue
    raise OSError("No writable report directory is available.")


def export_pipeline_report(
    main_window,
    filename: Optional[str] = None,
    format: str = "csv",
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports the current Top-K-filtered AnnoMASS results table to a result
    artifact the user can open in Excel or hand to a collaborator. Saved
    under the project-local reports/agent_reports directory."""
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}

    source_rows = int(len(df))
    df = _apply_result_filters(df, filters)
    if len(df) == 0:
        return {
            "error": "No AnnoMASS rows matched the requested filters.",
            "filters": filters or {},
            "source_rows": source_rows,
        }

    reports_dir = _writable_reports_dir()
    fmt = str(format or "csv").strip().lower()
    if fmt in ("excel", "xls"):
        fmt = "xlsx"
    if fmt not in ("csv", "xlsx", "md"):
        fmt = "csv"
    suffix = ".xlsx" if fmt == "xlsx" else f".{fmt}"
    name = (filename or "").strip() or f"pipeline_report_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"
    if Path(name).suffix.lower() not in (".csv", ".xlsx", ".md"):
        name += suffix
    # Never allow a path/traversal in the filename -- this always writes
    # inside _REPORTS_DIR, nowhere else on disk.
    name = Path(name).name
    out_path = reports_dir / name

    from core.result_table_export import compact_result_table, save_result_table
    export_df = compact_result_table(df)
    cols = list(export_df.columns)
    if fmt == "xlsx" or out_path.suffix.lower() == ".xlsx":
        save_result_table(export_df, out_path)
        actual_format = "xlsx"
    elif fmt == "md" or out_path.suffix.lower() == ".md":
        lines = ["# AnnoMASS Results", "", f"- Rows: {len(export_df)}", f"- Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        md_df = export_df.head(50).where(pd.notna(export_df.head(50)), "")
        md_cols = list(md_df.columns)
        def _md_cell(v):
            return str(v).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join(_md_cell(c) for c in md_cols) + " |")
        lines.append("| " + " | ".join("---" for _ in md_cols) + " |")
        for _, row in md_df.iterrows():
            lines.append("| " + " | ".join(_md_cell(row.get(c, "")) for c in md_cols) + " |")
        if len(export_df) > 50:
            lines.extend(["", f"_Showing first 50 of {len(export_df)} rows._"])
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        actual_format = "md"
    else:
        save_result_table(export_df, out_path)
        actual_format = "csv"

    try:
        stat_payload = _file_stat_payload(out_path)
        if not stat_payload["exists"] or not stat_payload["is_file"] or stat_payload["size_bytes"] <= 0:
            return {"error": f"Export verification failed for {out_path}", "file_path": str(out_path)}
        if actual_format == "csv":
            check_df = pd.read_csv(out_path)
            verified_rows = int(len(check_df))
        elif actual_format == "xlsx":
            check_df = pd.read_excel(out_path)
            verified_rows = int(len(check_df))
        else:
            verified_rows = int(len(export_df))
        if verified_rows != int(len(export_df)):
            return {
                "error": (
                    f"Export verification failed for {out_path}: expected "
                    f"{len(export_df)} rows, read back {verified_rows}"
                ),
                "file_path": str(out_path),
            }
    except Exception as exc:
        return {"error": f"Export verification failed for {out_path}: {exc}", "file_path": str(out_path)}

    return {
        "file_path": str(out_path),
        "n_rows": int(len(df)),
        "source_rows": source_rows,
        "filters": filters or {},
        "columns": cols,
        "format": actual_format,
        "verified": True,
        "verified_rows": verified_rows,
        "file_stat": stat_payload,
    }


def plot_pipeline_results(
    main_window,
    plot_type: str = "top1_scores",
    filename: Optional[str] = None,
    spectrum_index: Optional[int] = None,
    format: str = "pdf",
) -> Dict[str, Any]:
    """Generate a deterministic PNG figure from the current AnnoMASS results.
    This is intentionally template-based: the Agent can choose what evidence
    view to plot, but cannot execute arbitrary plotting code."""
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "deepmass2_matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"error": f"Plotting requires matplotlib, but it is unavailable: {exc}"}

    reports_dir = _writable_reports_dir()
    plot_type = str(plot_type or "top1_scores").strip().lower()
    fmt = str(format or "pdf").strip().lower()
    if fmt not in ("pdf", "png", "svg"):
        fmt = "pdf"
    allowed = {
        "evidence_distribution",
        "class_distribution",
        "top1_scores",
        "spectrum_candidates",
        "annosm_distribution",
    }
    if plot_type not in allowed:
        plot_type = "top1_scores"
    name = (filename or "").strip() or f"pipeline_plot_{plot_type}_{time.strftime('%Y%m%d_%H%M%S')}.{fmt}"
    suffix = Path(name).suffix.lower()
    if suffix not in (".pdf", ".png", ".svg"):
        name += f".{fmt}"
    out_path = reports_dir / Path(name).name

    work = df.copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)

    def _bar_counts(col: str, title: str, xlabel: str):
        if col not in work.columns:
            raise ValueError(f"Column not available: {col}")
        counts = work[col].fillna("(none)").astype(str).value_counts().sort_values(ascending=True)
        counts.plot(kind="barh", ax=ax, color="#2563EB")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        for i, v in enumerate(counts.values):
            ax.text(v, i, f" {int(v)}", va="center", fontsize=8)

    try:
        if plot_type == "evidence_distribution":
            _bar_counts("evidence_level", "AnnoMASS Evidence-Level Distribution", "Candidate rows")
        elif plot_type == "class_distribution":
            _bar_counts("flavonoid_class", "Flavonoid-Class Distribution", "Candidate rows")
        elif plot_type == "annosm_distribution":
            _bar_counts("annosm_state", "AnnoSM State Distribution", "Candidate rows")
        elif plot_type == "spectrum_candidates":
            if spectrum_index is None:
                if "spectrum_index" not in work.columns:
                    return {"error": "spectrum_index is required because results do not contain a spectrum_index column."}
                spectrum_index = int(work["spectrum_index"].iloc[0])
            if "spectrum_index" not in work.columns:
                return {"error": "Current results do not contain spectrum_index."}
            sub = work[work["spectrum_index"].astype(str) == str(int(spectrum_index))].copy()
            if sub.empty:
                return {"error": f"No AnnoMASS candidates found for spectrum_index={spectrum_index}."}
            if "spectrum_rank" in sub.columns:
                sub = sub.sort_values("spectrum_rank")
                rank_col = "spectrum_rank"
            else:
                sub = sub.sort_values("rank") if "rank" in sub.columns else sub
                rank_col = "rank" if "rank" in sub.columns else None
            score_cols = [c for c in ("level_score", "DreaMS_similarity", "library_cosine", "spectrum_predictor_score", "deepmass_score") if c in sub.columns]
            if not score_cols:
                return {"error": "No score columns available to plot."}
            labels = []
            for _, row in sub.head(12).iterrows():
                name_part = str(row.get("compound_name", "candidate"))[:28]
                rank_part = f"#{int(row[rank_col])} " if rank_col and pd.notna(row.get(rank_col)) else ""
                labels.append(rank_part + name_part)
            x = np.arange(len(labels))
            width = 0.8 / max(1, len(score_cols))
            for i, col in enumerate(score_cols):
                vals = pd.to_numeric(sub.head(12)[col], errors="coerce").fillna(0.0).clip(lower=0, upper=1)
                ax.bar(x + (i - (len(score_cols) - 1) / 2) * width, vals, width, label=col.replace("_", " "))
            ax.set_title(f"Candidate Evidence Scores for Spectrum {spectrum_index}")
            ax.set_ylabel("Score / similarity")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.legend(fontsize=8)
        else:
            top1 = _top1_rows_by_spectrum(work)
            if top1.empty:
                return {"error": "No Top-1 rows available to plot."}
            x_labels = [str(v) for v in top1.get("spectrum_index", range(len(top1)))]
            score_cols = [c for c in ("level_score", "DreaMS_similarity", "library_cosine", "spectrum_predictor_score", "deepmass_score") if c in top1.columns]
            if not score_cols:
                return {"error": "No score columns available to plot."}
            x = np.arange(len(top1))
            for col in score_cols:
                vals = pd.to_numeric(top1[col], errors="coerce")
                ax.plot(x, vals, marker="o", linewidth=1.8, label=col.replace("_", " "))
            ax.set_title("Top-1 Candidate Evidence Scores by Spectrum")
            ax.set_xlabel("Spectrum index")
            ax.set_ylabel("Score / similarity")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=0 if len(x_labels) <= 18 else 45, ha="center" if len(x_labels) <= 18 else "right")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", format=out_path.suffix.lower().lstrip("."))
        plt.close(fig)
        try:
            stat_payload = _file_stat_payload(out_path)
            if not stat_payload["exists"] or not stat_payload["is_file"] or stat_payload["size_bytes"] <= 0:
                return {"error": f"Plot verification failed for {out_path}", "file_path": str(out_path)}
        except Exception as exc:
            return {"error": f"Plot verification failed for {out_path}: {exc}", "file_path": str(out_path)}
        return {
            "file_path": str(out_path),
            "plot_type": plot_type,
            "format": out_path.suffix.lower().lstrip("."),
            "n_rows": int(len(work)),
            "spectrum_index": spectrum_index,
            "verified": True,
            "file_stat": stat_payload,
        }
    except Exception as exc:
        try:
            plt.close(fig)
        except Exception:
            pass
        return {"error": str(exc)}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _safe_num(value: Any, ndigits: int = 3) -> str:
    try:
        v = float(value)
        if v != v:
            return ""
        return f"{v:.{ndigits}f}"
    except Exception:
        return ""


def _top1_rows_by_spectrum(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "spectrum_index" not in df.columns:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()
    work = df.copy()
    if "spectrum_rank" in work.columns:
        rank_vals = pd.to_numeric(work["spectrum_rank"], errors="coerce")
        top = work[rank_vals.eq(1)].copy()
        if not top.empty:
            return top.sort_values("spectrum_index").reset_index(drop=True)
    sort_cols = [c for c in ("spectrum_index", "rank") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)
    return work.groupby("spectrum_index", as_index=False, sort=True).head(1).reset_index(drop=True)


def _add_doc_table(document, rows: List[Dict[str, Any]], columns: List[str], labels: Optional[Dict[str, str]] = None):
    labels = labels or {}
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, col in enumerate(columns):
        hdr[i].text = labels.get(col, col)
    for row in rows:
        cells = table.add_row().cells
        for i, col in enumerate(columns):
            val = row.get(col, "")
            if isinstance(val, float):
                cells[i].text = _safe_num(val, 4)
            else:
                cells[i].text = _safe_text(val)
    return table


def write_pipeline_word_report(
    main_window,
    filename: Optional[str] = None,
    title: str = "Flavonoid Identification Method and Results",
    include_top_n: int = 20,
) -> Dict[str, Any]:
    """Create a Word .docx method/results report from the current AnnoMASS
    results. This is deterministic evidence-grounded writing: it uses the
    result table and ranking cascade text, not hidden model guesses."""
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}

    try:
        import docx
    except Exception as exc:
        return {"error": f"Word export requires python-docx, but it is unavailable: {exc}"}

    reports_dir = _writable_reports_dir()
    name = (filename or "").strip() or f"pipeline_method_results_{time.strftime('%Y%m%d_%H%M%S')}.docx"
    if not name.lower().endswith(".docx"):
        name += ".docx"
    name = Path(name).name
    out_path = reports_dir / name

    df = df.copy()
    n_rows = int(len(df))
    n_spectra = int(df["spectrum_index"].nunique()) if "spectrum_index" in df.columns else None
    top1 = _top1_rows_by_spectrum(df)
    evidence_counts = (
        df["confidence_level"].fillna("(none)").astype(str).value_counts().to_dict()
        if "confidence_level" in df.columns else {}
    )

    document = docx.Document()
    document.add_heading(title or "Flavonoid Identification Method and Results", level=0)
    document.add_paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    document.add_heading("Methods", level=1)
    document.add_paragraph(
        "Positive-ion MS/MS spectra were annotated with a staged waterfall. Each spectrum "
        "was first searched against the DreaMS flavonoid reference library (level A); "
        "spectra without a passing DreaMS candidate went to DeepMASS structure candidates, "
        "first scored against their experimental library spectra when available (level B); "
        "if no B candidate passed, every candidate was scored against its ICEBERG-predicted "
        "spectrum (level C); spectra still unannotated "
        "went to the DeepMet CLM library-external route (level D). AnnoSM, when available, "
        "was used as a gated isomer-resolution check inside a level rather than a free-form score."
    )
    document.add_paragraph(DEEPMET_ROUTE_EXPLANATION_TEXT)

    document.add_heading("Transparent Selection Rule", level=1)
    document.add_paragraph(CASCADE_EXPLANATION_TEXT)
    document.add_paragraph(
        "The displayed rank is traceable from raw evidence columns: confidence_level "
        "(A-D), msi_level, level_score and the two component scores of that level "
        "(DreaMS_similarity / library_cosine / deepmass_score / spectrum_predictor_score). "
        "Levels map to MSI confidence as A and B = Level 2 (experimental reference "
        "spectrum), C = Level 3 (in-silico prediction plus structure prior), D = Level 4 "
        "(generated structure, no reference)."
    )

    document.add_heading("Results Summary", level=1)
    summary_lines = [f"Candidate rows: {n_rows}"]
    if n_spectra is not None:
        summary_lines.append(f"Spectra with results: {n_spectra}")
    summary_lines.append(
        "Annotation-level distribution: " + (
            ", ".join(f"{k}: {v}" for k, v in evidence_counts.items()) if evidence_counts else "not available"
        )
    )
    for line in summary_lines:
        document.add_paragraph(line, style=None)

    if top1 is not None and not top1.empty:
        document.add_heading("Top-1 Candidate Per Spectrum", level=2)
        cols = [
            "spectrum_index", "spectrum_rank", "rank", "compound_name",
            "confidence_level", "msi_level", "level_score", "candidate_source", "formula",
            "DreaMS_similarity", "library_cosine", "deepmass_score", "spectrum_predictor_score",
        ]
        cols = [c for c in cols if c in top1.columns]
        rows = _safe_records(top1, cols, limit=None)
        _add_doc_table(document, rows, cols, {
            "spectrum_index": "Spectrum",
            "spectrum_rank": "Spectrum rank",
            "rank": "Global rank",
            "compound_name": "Candidate",
            "confidence_level": "Level",
            "msi_level": "MSI",
            "level_score": "Level score",
            "candidate_source": "Source",
            "formula": "Formula",
            "DreaMS_similarity": "DreaMS",
            "library_cosine": "Library cosine",
            "deepmass_score": "DeepMASS",
            "spectrum_predictor_score": "Predicted spectrum",
        })

    document.add_heading("Candidate Evidence Table", level=2)
    top_n = max(1, int(include_top_n or 20))
    cols = [
        "rank", "spectrum_rank", "compound_name", "confidence_level", "msi_level", "level_score",
        "candidate_source", "formula", "flavonoid_class", "DreaMS_similarity", "library_cosine",
        "deepmass_score", "spectrum_predictor_score", "annosm_state", "annosm_score", "spectrum_index",
    ]
    cols = [c for c in cols if c in df.columns]
    rows = _safe_records(df, cols, limit=top_n)
    _add_doc_table(document, rows, cols, {
        "rank": "Global rank",
        "spectrum_rank": "Spectrum rank",
        "compound_name": "Candidate",
        "confidence_level": "Level",
        "msi_level": "MSI",
        "level_score": "Level score",
        "candidate_source": "Source",
        "formula": "Formula",
        "flavonoid_class": "Class",
        "DreaMS_similarity": "DreaMS",
        "library_cosine": "Library cosine",
        "deepmass_score": "DeepMASS",
        "spectrum_predictor_score": "Predicted spectrum",
        "annosm_state": "Substituent-check state",
        "annosm_score": "Substituent check",
        "spectrum_index": "Spectrum",
    })

    document.add_heading("Interpretation Notes", level=1)
    document.add_paragraph(
        "Levels A and B rest on an experimental reference spectrum; level C rests on a "
        "DeepMASS structure prior plus an ICEBERG-predicted spectrum and is less certain; "
        "level D is a generated structure without any reference. AnnoSM states and scores are "
        "diagnostic fragment-compatibility annotations; AnnoSM changes the displayed "
        "rank only when annosm_promoted is true. "
        "Retention-time predictions, if present, are system-dependent and should be "
        "treated as supporting evidence only."
    )

    document.save(out_path)
    try:
        stat_payload = _file_stat_payload(out_path)
        if not stat_payload["exists"] or not stat_payload["is_file"] or stat_payload["size_bytes"] <= 0:
            return {"error": f"Word report verification failed for {out_path}", "file_path": str(out_path)}
    except Exception as exc:
        return {"error": f"Word report verification failed for {out_path}: {exc}", "file_path": str(out_path)}
    return {
        "file_path": str(out_path),
        "n_rows": n_rows,
        "n_spectra": n_spectra,
        "top1_rows": int(len(top1)) if top1 is not None else 0,
        "format": "docx",
        "include_top_n": top_n,
        "verified": True,
        "file_stat": stat_payload,
    }


def _categorize_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Groups a candidate record's already-existing evidence fields into 3
    categories. Purely organizational -- no new value is computed here and
    nothing feeds back into ranking. "Conflicting evidence" (candidates
    disagreeing with each other) is not a single-candidate concept -- see the
    compare_isomers tool for that."""
    has_rt = record.get("predicted_rt_min") is not None
    return {
        "spectrum_evidence": {
            "annotation_level": record.get("confidence_level"),
            "msi_level": record.get("msi_level"),
            "level_score": record.get("level_score"),
            "dreams_similarity": record.get("DreaMS_similarity"),
            "library_spectrum_cosine": record.get("library_cosine"),
            "deepmass_score": record.get("deepmass_score"),
        },
        "fragment_evidence": {
            "spectrum_predictor_score": record.get("spectrum_predictor_score"),
            "iceberg_entropy_similarity": record.get("iceberg_entropy_similarity"),
            "iceberg_cosine_similarity": record.get("iceberg_cosine_similarity"),
            "annosm_state": record.get("annosm_state"),
            "annosm_score": record.get("annosm_score"),
            "annosm_promoted": record.get("annosm_promoted"),
            "annosm_changed_rank": bool(record.get("annosm_promoted") is True),
            "annosm_note": (
                "AnnoSM changed the rank only if annosm_promoted is true; "
                "state/score alone are diagnostic annotations."
            ),
        },
        "orthogonal_evidence": {
            "predicted_rt_min": record.get("predicted_rt_min"),
            "note": (
                "Retention time is system-dependent (column/mobile phase) -- "
                "treat as supporting, not definitive, evidence."
            ) if has_rt else "No retention-time evidence available for this candidate.",
        },
    }


_LEVEL_FORMULA = {
    "A": ("DreaMS similarity", "DreaMS_similarity", "library-spectrum cosine", "library_cosine"),
    "B": ("DeepMASS score", "deepmass_score", "library-spectrum cosine", "library_cosine"),  # + DreaMS_similarity
    "C": ("DeepMASS score", "deepmass_score", "ICEBERG predicted-spectrum similarity",
          "spectrum_predictor_score"),
    "D": ("ICEBERG entropy similarity", "iceberg_entropy_similarity",
          "ICEBERG cosine similarity", "iceberg_cosine_similarity"),
}


def _selection_trace(record: Dict[str, Any]) -> Dict[str, Any]:
    level = str(record.get("confidence_level") or "")
    first_name, first_col, second_name, second_col = _LEVEL_FORMULA.get(
        level, ("first score", "", "second score", ""))
    first = record.get(first_col) if first_col else None
    second = record.get(second_col) if second_col else None
    if level == "D":
        rule = ("kept as level D because entropy and cosine similarity both >= 0.5 and "
                "both pick this structure as Top-1 for the spectrum")
    elif level == "B":
        rule = ("level score = 0.6 x DeepMASS score + 0.1 x library-spectrum cosine + 0.3 x the DreaMS "
                "similarity carried down from level A (re-weighted over DeepMASS + cosine when level A was not run), "
                "compared with the fixed level-B threshold 0.97")
    elif level == "C":
        rule = ("level score = 0.3 x the candidate's carried level-B score (DeepMASS + "
                "library cosine + DreaMS) + 0.7 x the ICEBERG predicted-spectrum similarity, compared "
                "with the fixed level-C threshold 0.12")
    else:
        rule = (f"level score = weighted sum of {first_name} and {second_name}"
                + (" (0.8 / 0.2, per-compound pooled over its library hits), compared with the fixed level-A threshold 0.70" if level == "A"
                   else f", compared with the fixed level-{level or '?'} threshold"))
    return {
        "global_rank": record.get("rank"),
        "spectrum_rank": record.get("spectrum_rank"),
        "waterfall": [
            "A: DreaMS library match (DreaMS similarity + library-spectrum cosine)",
            "B: DeepMASS candidates + DreaMS hits that failed A (DeepMASS + library cosine + DreaMS)",
            "C: every candidate, including those that failed B (carried B score + ICEBERG similarity)",
            "D: DeepMet CLM structure (ICEBERG entropy + cosine dual Top-1)",
        ],
        "values_used": {
            "annotation_level": level,
            "msi_level": record.get("msi_level"),
            "level_score": record.get("level_score"),
            "modules_run": record.get("modules_run"),
            first_name: first,
            second_name: second,
            "annosm_state": record.get("annosm_state"),
            "annosm_score": record.get("annosm_score"),
        },
        "plain_language": (
            f"The spectrum stopped at level {level or '?'} (the first level with a passing "
            f"candidate); {rule}. {first_name} {_safe_num(first, 3) or 'n/a'}, "
            f"{second_name} {_safe_num(second, 3) or 'n/a'}, level score "
            f"{_safe_num(record.get('level_score'), 3) or 'n/a'}. Candidates of this spectrum "
            f"are ordered by level score."
        ),
        "annosm_caution": (
            "annosm_state/annosm_score are diagnostic annotations, not proof that "
            "AnnoSM changed the rank. Only annosm_promoted=True means AnnoSM "
            "actually promoted a candidate in the gated rerank."
        ),
    }


def explain_pipeline_results(main_window, spectrum_index: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        df = getattr(main_window, "_pipeline_last_df_full", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}
    df = df.copy()
    if spectrum_index is not None and "spectrum_index" in df.columns:
        df = df[df["spectrum_index"] == spectrum_index]
        if df.empty:
            return {"error": f"No results found for spectrum_index={spectrum_index}."}
    records = _safe_records(df, _PIPELINE_RESULT_COLUMNS, limit=top_k)
    for r in records:
        r["evidence"] = _categorize_evidence(r)
        r["selection_trace"] = _selection_trace(r)
    return {
        "cascade_explanation": CASCADE_EXPLANATION_TEXT,
        "transparency_note": (
            "Selection is auditable from returned columns and selection_trace; "
            "no hidden LLM reasoning or learned composite probability is used for final ranking."
        ),
        "candidates": records,
    }


def compare_isomers(main_window, rank_a: int, rank_b: int, spectrum_index: Optional[int] = None) -> Dict[str, Any]:
    """Structured side-by-side comparison of two already-ranked candidates
    (by their `rank` field, e.g. Top-1 vs Top-2 when they look like isomers)
    from the last AnnoMASS run. Does NOT compute a new composite 'winner'
    score -- that would violate the ranking boundary (see module docstring).
    Shows each candidate's own categorized evidence side by side; the
    EXISTING rank (the real cascade's actual output) is left to stand as the
    preference signal. This tool only makes *why* legible, it doesn't
    re-decide anything."""
    df = getattr(main_window, "_pipeline_last_df", None)
    if df is None or len(df) == 0:
        df = getattr(main_window, "_pipeline_last_df_full", None)
    if df is None or len(df) == 0:
        return {"error": "AnnoMASS has not been run yet (no results available)."}
    df = df.copy()
    if spectrum_index is not None and "spectrum_index" in df.columns:
        df = df[df["spectrum_index"] == spectrum_index]
    if "rank" not in df.columns:
        return {"error": "No 'rank' column in the current AnnoMASS results."}

    row_a = df[df["rank"] == rank_a]
    row_b = df[df["rank"] == rank_b]
    if row_a.empty or row_b.empty:
        return {"error": f"Could not find both rank {rank_a} and rank {rank_b} in the current results."}

    rec_a = _safe_records(row_a, _PIPELINE_RESULT_COLUMNS, limit=1)[0]
    rec_b = _safe_records(row_b, _PIPELINE_RESULT_COLUMNS, limit=1)[0]
    rec_a["evidence"] = _categorize_evidence(rec_a)
    rec_b["evidence"] = _categorize_evidence(rec_b)

    formula_a, formula_b = rec_a.get("formula"), rec_b.get("formula")
    same_formula = bool(formula_a) and formula_a == formula_b

    diagnostic_fragments = _isomer_diagnostic_fragments(main_window, rec_a, rec_b) if same_formula else None

    return {
        "candidate_a": rec_a,
        "candidate_b": rec_b,
        "same_molecular_formula": same_formula,
        "diagnostic_fragments": diagnostic_fragments,
        "note": (
            "Same molecular formula -- likely positional/glycosidic isomers; "
            "MS2 alone may not fully resolve them. annosm_state/annosm_score "
            "in each candidate's fragment_evidence is AnnoSM's aggregate "
            "verdict; diagnostic_fragments (if present) breaks that down "
            "into the actual CFIL fragment ion(s) -- m/z and matched "
            "intensity -- that support each side's substitution mode, so "
            "you can cite the specific ion evidence instead of just the "
            "score."
        ) if same_formula else "Different molecular formulas -- not classic isomers of each other.",
    }


def _spectrum_peaks(main_window, spectrum_index) -> Optional[Any]:
    """(N, 2) [mz, intensity] array for one loaded spectrum, in the same
    shape AnnoSM's own pipeline rerank builds (see annomass_app.py's
    Step 4.5). Returns None if unavailable rather than raising -- this is
    explanatory/optional evidence, never required for compare_isomers to
    work."""
    spectrums = getattr(main_window, "spectrums", None)
    if spectrums is None or spectrum_index is None:
        return None
    try:
        sp = spectrums.iloc[int(spectrum_index)]["spectrum"]
        return np.column_stack([
            np.asarray(sp.peaks.mz, dtype=float),
            np.asarray(sp.peaks.intensities, dtype=float),
        ])
    except Exception:
        return None


def _isomer_diagnostic_fragments(main_window, rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """For two same-formula candidates, surfaces the specific CFIL
    diagnostic fragment ions (m/z + matched intensity) behind each side's
    annosm_score -- see core/annosm_bridge.py's diagnostic_fragment_detail.
    Purely explanatory/on-demand (recomputed here, not read back from the
    pipeline run); never touches ranking. Returns None if AnnoSM, a SMILES,
    or the underlying spectrum isn't available for either side."""
    if not annosm_bridge.is_available():
        return None
    try:
        preset = annosm_bridge.ANNOSM_LIBRARIES[annosm_bridge.PIPELINE_RERANK_PRESET]
        annotator = annosm_bridge.get_annotator(preset["cfil"])
    except Exception:
        return None

    out = {}
    for key, rec in (("candidate_a", rec_a), ("candidate_b", rec_b)):
        candidate_sm = annosm_bridge.candidate_sm_from_smiles(rec.get("smiles"))
        if candidate_sm is None:
            continue
        peaks = _spectrum_peaks(main_window, rec.get("spectrum_index"))
        if peaks is None:
            continue
        matches = annosm_bridge.diagnostic_fragment_detail(
            annotator, peaks,
            substitution_modes=[candidate_sm.annotated_sm, candidate_sm.raw_sm],
            oh_sum=candidate_sm.oh_sum, och3_sum=candidate_sm.och3_sum,
        )
        if matches:
            out[key] = {
                "substitution_mode": candidate_sm.annotated_sm,
                "matched_fragments": matches,
            }
    return out or None


def query_structure_database(main_window, query: str, field: str = "auto", limit: int = 20) -> Dict[str, Any]:
    database = getattr(main_window, "default_database", None)
    if database is None or len(database) == 0:
        return {"error": "Structure database is not loaded yet."}
    hits = structure_db_query.query_structure_db(database, query, field=field, limit=limit)
    return {"query": query, "n_hits": len(hits), "hits": hits}


def query_reference_spectra(main_window, query: str, field: str = "auto", source: str = "all", limit: int = 20) -> Dict[str, Any]:
    hits = flavonoid_reference_db.query_reference_spectra(query, field=field, source=source, limit=limit)
    return {"query": query, "n_hits": len(hits), "hits": hits}


def run_abcort_predict(main_window, smiles_list: List[str], checkpoint: Optional[str] = None) -> Dict[str, Any]:
    if not abcort_bridge.is_available():
        return {"error": f"ABCoRT unavailable: {abcort_bridge.unavailable_reason()}"}
    try:
        results = abcort_bridge.predict_rt_batch(smiles_list, checkpoint_path=checkpoint)
    except Exception as exc:
        return {"error": str(exc)}
    return {"predictions": results}


def remember_fact(main_window, fact: str, category: str = "general") -> Dict[str, Any]:
    """Persist a fact to ~/.annomass/agent_memory.md, loaded back into the
    system prompt at the start of every future conversation (see
    AgentWorker.run() in annomass_app.py). Pure file I/O, no GUI needed."""
    if not fact or not fact.strip():
        return {"error": "Empty fact, nothing saved."}
    agent_memory.append_memory(fact.strip(), category=category or "general")
    return {"saved": True}


def list_agent_skills(main_window) -> Dict[str, Any]:
    skills = agent_skills.list_skills()
    return {"n_skills": len(skills), "skills": skills}


def install_agent_skill(
    main_window,
    name: str,
    content: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    return agent_skills.install_skill(name=name, content=content, source_path=source_path)


def delete_agent_skill(main_window, name: str) -> Dict[str, Any]:
    return agent_skills.delete_skill(name=name)


def query_massql(main_window, query: str, limit: int = 100) -> Dict[str, Any]:
    """Run the local MassQL subset against spectra currently loaded in the GUI."""
    if main_window is None or not hasattr(main_window, "spectrums") or len(main_window.spectrums) == 0:
        return {"error": "No spectra loaded. Load an MGF/MSP/MAT file first."}
    spectra = list(main_window.spectrums["spectrum"])
    result = massql.run_query(spectra, query, limit=limit)
    return {
        "query": query,
        "n_matches": result.get("n_matches", 0),
        "matches": result.get("matches", [])[:limit],
        "note": result.get("note"),
    }


def query_massql_file(main_window, query: str, file_path: str, limit: int = 100) -> Dict[str, Any]:
    """Run the official MassQL engine on a local MS data/library file."""
    try:
        return massql.run_official_file_query(file_path, query, limit=limit)
    except Exception as exc:
        return {"error": str(exc)}


def query_massql_remote_file(main_window, query: str, source_url: str, limit: int = 100, max_mb: int = 300) -> Dict[str, Any]:
    """Download a direct remote MS data URL and run the official MassQL engine."""
    try:
        return massql.run_online_query(source_url, query, limit=limit, max_mb=max_mb)
    except Exception as exc:
        return {"error": str(exc)}


def query_massql_gnps(main_window, query: str) -> Dict[str, Any]:
    """Use GNPS2 MassQL online service for parsing/workflow link generation."""
    try:
        return massql.query_gnps_massql_service(query)
    except Exception as exc:
        return {"error": str(exc)}


def query_massql_libraries(main_window, query: str, libraries: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prepare multi-library online MassQL routes for interpretation."""
    try:
        return massql.query_online_library_router(query, libraries=libraries)
    except Exception as exc:
        return {"error": str(exc)}


def query_pubchem(main_window, query: str, field: str = "auto", limit: int = 5) -> Dict[str, Any]:
    """Live PubChem lookup for a compound not found in the local structure DB."""
    return pubchem_lookup.query_pubchem(query, field=field, limit=limit)


def search_literature(main_window, query: str, max_results: int = 5) -> Dict[str, Any]:
    """Live PubMed literature search, e.g. to support explaining a compound/spectrum result."""
    return pubmed_search.search_pubmed(query, max_results=max_results)


def query_massbank_online(main_window, query: str, field: str = "auto", limit: int = 10) -> Dict[str, Any]:
    """Live MassBank lookup (real MS/MS records, not just structure metadata) for a compound
    beyond the local 154-compound curated MassBank standard-reference subset."""
    return massbank_lookup.query_massbank(query, field=field, limit=limit)


def query_mona_online(main_window, query: str, field: str = "auto", limit: int = 10) -> Dict[str, Any]:
    """Live MoNA lookup for external MS/MS records or actionable MoNA links."""
    return mona_lookup.query_mona(query, field=field, limit=limit)


# ---------------------------------------------------------------------------
# Project code tools: Codex-lite, but workspace-scoped and command-whitelisted
# ---------------------------------------------------------------------------

def _project_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except Exception:
        return str(path)


def _resolve_project_path(path: str, *, for_write: bool = False, create: bool = False) -> Path:
    if not path or not str(path).strip():
        raise ValueError("path is required")
    raw = Path(str(path)).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (_PROJECT_ROOT / raw).resolve()
    if resolved != _PROJECT_ROOT and _PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"path escapes project root: {path}")
    parts = set(resolved.relative_to(_PROJECT_ROOT).parts)
    blocked = sorted(parts & _CODE_EDIT_DENY_PARTS)
    if blocked:
        raise ValueError(f"path is in a protected project area: {blocked[0]}")
    if for_write:
        if resolved.exists() and resolved.is_dir():
            raise ValueError("cannot write a directory")
        suffix = resolved.suffix.lower()
        if suffix not in _CODE_EDIT_ALLOWED_SUFFIXES:
            raise ValueError(f"file suffix '{suffix or '(none)'}' is not allowed for code-agent writes")
        if not create and not resolved.exists():
            raise ValueError("file does not exist; pass create=true to create it")
    return resolved


def read_project_file(main_window, path: str, max_chars: int = 30000) -> Dict[str, Any]:
    """Read a text/code file inside the project workspace."""
    try:
        p = _resolve_project_path(path)
        if not p.exists() or not p.is_file():
            return {"error": f"File not found: {path}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        max_chars = max(1000, min(int(max_chars or 30000), 120000))
        truncated = len(text) > max_chars
        return {
            "path": _project_relpath(p),
            "chars": len(text),
            "truncated": truncated,
            "content": text[:max_chars],
        }
    except Exception as exc:
        return {"error": str(exc)}


def list_project_files(main_window, root: str = ".", file_glob: str = "", max_files: int = 200) -> Dict[str, Any]:
    """List project files while skipping protected/heavy directories."""
    try:
        base = _resolve_project_path(root or ".")
        if not base.exists():
            return {"error": f"Path not found: {root}"}
        max_files = max(1, min(int(max_files or 200), 1000))
        files = []
        iterator = base.rglob("*") if base.is_dir() else [base]
        for p in iterator:
            if len(files) >= max_files:
                break
            try:
                rel = p.relative_to(_PROJECT_ROOT)
            except Exception:
                continue
            if set(rel.parts) & _CODE_EDIT_DENY_PARTS:
                continue
            if not p.is_file():
                continue
            if file_glob and not p.match(file_glob):
                continue
            files.append({
                "path": str(rel),
                "size": int(p.stat().st_size),
                "suffix": p.suffix.lower(),
            })
        return {"root": _project_relpath(base), "n_files": len(files), "files": files}
    except Exception as exc:
        return {"error": str(exc)}


def search_project(main_window, query: str, file_glob: str = "", max_matches: int = 80,
                   literal: bool = True) -> Dict[str, Any]:
    """Search project text/code files with rg, falling back to Python search."""
    query = str(query or "")
    if not query:
        return {"error": "query is required"}
    max_matches = max(1, min(int(max_matches or 80), 200))
    matches: List[Dict[str, Any]] = []
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color", "never",
            "--glob", "!data/**", "--glob", "!model/**", "--glob", "!.git/**",
            "--glob", "!__pycache__/**",
        ]
        if literal:
            cmd.append("--fixed-strings")
        if file_glob:
            cmd.extend(["--glob", str(file_glob)])
        cmd.extend([query, str(_PROJECT_ROOT)])
        proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20)
        if proc.returncode not in (0, 1):
            return {"error": proc.stderr.strip() or f"rg failed with exit code {proc.returncode}"}
        for line in proc.stdout.splitlines():
            if len(matches) >= max_matches:
                break
            try:
                file_part, line_no, text = line.split(":", 2)
                p = Path(file_part).resolve()
                matches.append({
                    "path": _project_relpath(p),
                    "line": int(line_no),
                    "text": text[:500],
                })
            except Exception:
                matches.append({"path": "", "line": None, "text": line[:500]})
        return {"query": query, "n_matches": len(matches), "matches": matches}
    except FileNotFoundError:
        pass
    except Exception as exc:
        return {"error": str(exc)}

    try:
        for p in _PROJECT_ROOT.rglob("*"):
            if len(matches) >= max_matches:
                break
            if not p.is_file() or p.suffix.lower() not in _CODE_EDIT_ALLOWED_SUFFIXES:
                continue
            rel_parts = set(p.relative_to(_PROJECT_ROOT).parts)
            if rel_parts & _CODE_EDIT_DENY_PARTS:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), start=1):
                haystack = line if literal else line.lower()
                needle = query if literal else query.lower()
                if needle in haystack:
                    matches.append({"path": _project_relpath(p), "line": line_no, "text": line[:500]})
                    if len(matches) >= max_matches:
                        break
        return {"query": query, "n_matches": len(matches), "matches": matches, "fallback": "python"}
    except Exception as exc:
        return {"error": str(exc)}


def write_project_file(main_window, path: str, content: str, expected_current_content: str = "",
                       create: bool = False) -> Dict[str, Any]:
    """Write a text/code file inside the project workspace with guardrails."""
    try:
        p = _resolve_project_path(path, for_write=True, create=bool(create))
        content = str(content if content is not None else "")
        if len(content) > _MAX_CODE_WRITE_CHARS:
            return {"error": f"content too large ({len(content)} chars); max is {_MAX_CODE_WRITE_CHARS}"}
        existed = p.exists()
        old = p.read_text(encoding="utf-8", errors="replace") if existed else ""
        if expected_current_content and old != expected_current_content:
            return {
                "error": "current file content differs from expected_current_content; read the file again before writing",
                "path": _project_relpath(p),
            }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        if not p.exists() or not p.is_file():
            return {"error": f"write verification failed; file is not present on disk: {_project_relpath(p)}"}
        disk_text = p.read_text(encoding="utf-8", errors="replace")
        if disk_text != content:
            return {
                "error": "write verification failed; disk content differs from requested content",
                "path": _project_relpath(p),
                "expected_chars": len(content),
                "actual_chars": len(disk_text),
                "verified": False,
            }
        stat_payload = _file_stat_payload(p)
        return {
            "path": _project_relpath(p),
            "file_path": str(p),
            "changed_files": [_project_relpath(p)],
            "created": not existed,
            "old_chars": len(old),
            "new_chars": len(content),
            "verified": True,
            "file_stat": stat_payload,
            "note": "Run get_project_diff and an allowed validation command next.",
        }
    except Exception as exc:
        return {"error": str(exc)}


def _paths_from_unified_patch(patch_text: str) -> List[str]:
    paths = []
    for line in patch_text.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].strip().split("\t", 1)[0].split(" ", 1)[0]
        if raw == "/dev/null":
            paths.append(raw)
            continue
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        if raw:
            paths.append(raw)
    return paths


def apply_project_patch(main_window, patch: str, check_only: bool = False) -> Dict[str, Any]:
    """Apply a unified diff patch after validating every touched path."""
    try:
        patch = str(patch or "")
        if not patch.strip():
            return {"error": "patch is required"}
        if "GIT binary patch" in patch or "\x00" in patch:
            return {"error": "binary patches are not allowed"}
        raw_paths = _paths_from_unified_patch(patch)
        if not raw_paths:
            return {"error": "No unified-diff file headers found in patch."}
        if any(p == "/dev/null" for p in raw_paths[1::2]):
            return {"error": "Deleting files via apply_project_patch is not allowed."}
        touched = []
        for raw in sorted(set(p for p in raw_paths if p != "/dev/null")):
            # `create=True` permits new files while still enforcing suffix and
            # protected-directory rules.
            p = _resolve_project_path(raw, for_write=True, create=True)
            touched.append(_project_relpath(p))
        tmp_dir = Path("/tmp") / "deepmass2_agent_patches"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patch_path = tmp_dir / f"agent_patch_{int(time.time() * 1000)}.diff"
        patch_path.write_text(patch, encoding="utf-8")
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)],
            cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20,
        )
        if check.returncode != 0:
            return {
                "error": check.stderr.strip() or f"git apply --check failed with exit code {check.returncode}",
                "touched_files": touched,
            }
        if check_only:
            return {"ok": True, "check_only": True, "touched_files": touched, "changed_files": []}
        apply = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20,
        )
        if apply.returncode != 0:
            return {
                "error": apply.stderr.strip() or f"git apply failed with exit code {apply.returncode}",
                "touched_files": touched,
            }
        verified_files = []
        for rel in touched:
            p = _resolve_project_path(rel)
            if not p.exists() or not p.is_file():
                return {
                    "error": f"patch verification failed; touched file is not present on disk: {rel}",
                    "touched_files": touched,
                    "verified": False,
                }
            verified_files.append(_file_stat_payload(p))
        return {
            "ok": True,
            "touched_files": touched,
            "changed_files": touched,
            "verified": True,
            "verified_files": verified_files,
            "note": "Run get_project_diff and an allowed validation command next.",
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_project_diff(main_window, path: str = "", max_chars: int = 40000) -> Dict[str, Any]:
    """Return git diff for the project or one file."""
    try:
        cmd = ["git", "diff", "--"]
        if path:
            p = _resolve_project_path(path)
            cmd.append(str(p.relative_to(_PROJECT_ROOT)))
        proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or f"git diff failed with exit code {proc.returncode}"}
        diff = proc.stdout
        max_chars = max(2000, min(int(max_chars or 40000), 120000))
        return {"chars": len(diff), "truncated": len(diff) > max_chars, "diff": diff[:max_chars]}
    except Exception as exc:
        return {"error": str(exc)}


def summarize_project_changes(main_window, max_chars: int = 40000) -> Dict[str, Any]:
    """Return a compact Codex-style summary payload for recent project edits."""
    try:
        diff_result = get_project_diff(main_window, max_chars=max_chars)
        if diff_result.get("error"):
            return diff_result
        changed = []
        proc = subprocess.run(
            ["git", "diff", "--name-only", "--"],
            cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20,
        )
        if proc.returncode == 0:
            changed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return {
            "changed_files": changed,
            "n_changed_files": len(changed),
            "diff_chars": diff_result.get("chars", 0),
            "diff_truncated": diff_result.get("truncated", False),
            "diff": diff_result.get("diff", ""),
            "final_reply_template": (
                "Use a concise Codex-style final answer: changed files, what changed, "
                "validation run/pass-fail, and remaining risk."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _iter_review_files(target: str, scope: str, max_files: int) -> List[Path]:
    files: List[Path] = []
    seen = set()

    def add_path(p: Path):
        try:
            p = p.resolve()
            if p in seen:
                return
            rel = p.relative_to(_PROJECT_ROOT)
            if set(rel.parts) & _CODE_EDIT_DENY_PARTS:
                return
            if p.is_file() and p.suffix.lower() in _CODE_EDIT_ALLOWED_SUFFIXES:
                seen.add(p)
                files.append(p)
        except Exception:
            return

    if target:
        base = _resolve_project_path(target)
        if base.is_file():
            add_path(base)
        elif base.is_dir():
            for p in base.rglob("*"):
                if len(files) >= max_files:
                    break
                add_path(p)
        return files[:max_files]

    if scope == "changed":
        proc = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRT", "--"],
            cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if len(files) >= max_files:
                    break
                add_path(_PROJECT_ROOT / line.strip())
        other = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(_PROJECT_ROOT), text=True, capture_output=True, timeout=20,
        )
        if other.returncode == 0:
            for line in other.stdout.splitlines():
                if len(files) >= max_files:
                    break
                add_path(_PROJECT_ROOT / line.strip())
    else:
        for p in _PROJECT_ROOT.rglob("*"):
            if len(files) >= max_files:
                break
            add_path(p)
    return files[:max_files]


def review_project_code(main_window, target: str = "", scope: str = "changed",
                        max_files: int = 80, max_findings: int = 80) -> Dict[str, Any]:
    """Lightweight code self-review for the Agent: syntax plus static risk checks."""
    try:
        scope = str(scope or "changed").strip().lower()
        if scope not in ("changed", "all"):
            scope = "changed"
        max_files = max(1, min(int(max_files or 80), 300))
        max_findings = max(1, min(int(max_findings or 80), 300))
        files = _iter_review_files(target, scope, max_files)
        findings: List[Dict[str, Any]] = []

        def add(severity: str, path: Path, line: Optional[int], message: str, snippet: str = ""):
            if len(findings) >= max_findings:
                return
            findings.append({
                "severity": severity,
                "path": _project_relpath(path),
                "line": line,
                "message": message,
                "snippet": snippet[:300],
            })

        sensitive_markers = (
            "core/staged_levels.py",
            "rank_key",
            "_rank_key",
            "assign_levels",
            "_pipeline_last_df",
        )
        risky_patterns = [
            ("high", "shell=True", "subprocess call uses shell=True; avoid shell execution in GUI/Agent code"),
            ("high", "os.system(", "os.system executes through a shell; use whitelisted argv subprocess helpers"),
            ("high", "eval(", "eval on dynamic input is unsafe"),
            ("high", "exec(", "exec on dynamic input is unsafe"),
            ("high", "git reset --hard", "destructive git reset should not be reachable from the Agent"),
            ("high", "rm -rf", "destructive recursive delete should not be reachable from the Agent"),
            ("medium", "pickle.load", "pickle.load executes trusted Python object graphs; ensure the source is trusted"),
            ("medium", "verify=False", "HTTP TLS verification disabled"),
            ("medium", "yaml.load(", "yaml.load can be unsafe unless SafeLoader is used"),
            ("medium", "except Exception:", "broad exception handler; verify errors are surfaced where correctness matters"),
            ("low", "pass  #", "suppressed block; verify this is intentionally non-fatal"),
            ("low", "TODO", "TODO left in code"),
            ("low", "FIXME", "FIXME left in code"),
        ]

        for p in files:
            rel = _project_relpath(p)
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                add("medium", p, None, f"could not read file: {exc}")
                continue
            if len(text) > _MAX_CODE_WRITE_CHARS:
                add("low", p, None, f"large file skipped for detailed scan ({len(text)} chars)")
                continue
            if any(line.startswith(("<<<<<<< ", "=======", ">>>>>>> ")) for line in text.splitlines()):
                add("high", p, None, "possible unresolved merge-conflict marker")
            if rel == "core/staged_levels.py" or any(marker in text for marker in sensitive_markers):
                add("medium", p, None, "ranking/scientific evidence-sensitive code touched or referenced; review selection logic carefully")
            if p.suffix.lower() == ".py":
                try:
                    py_compile.compile(str(p), doraise=True)
                except py_compile.PyCompileError as exc:
                    add("high", p, None, f"Python syntax/compile error: {exc.msg}")
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                for severity, pattern, message in risky_patterns:
                    if pattern in line:
                        # Avoid nagging on explicit explanatory text in this review helper itself.
                        if rel == "core/agent_tools.py" and "risky_patterns" in text[max(0, text.find(line)-200):text.find(line)+200]:
                            continue
                        add(severity, p, line_no, message, stripped)
                if "subprocess.run(" in line and "shell=" not in line:
                    # Informational reminder for command paths in Agent-facing code.
                    if "agent" in rel.lower() or rel.startswith("core/"):
                        add("low", p, line_no, "subprocess invocation; verify argv whitelist, cwd, timeout, and captured output", stripped)

        severity_order = {"high": 0, "medium": 1, "low": 2}
        findings.sort(key=lambda f: (severity_order.get(f["severity"], 9), f["path"], f["line"] or 0))
        counts = {sev: sum(1 for f in findings if f["severity"] == sev) for sev in ("high", "medium", "low")}
        return {
            "scope": scope,
            "target": target or None,
            "n_files_reviewed": len(files),
            "files_reviewed": [_project_relpath(p) for p in files],
            "n_findings": len(findings),
            "severity_counts": counts,
            "findings": findings[:max_findings],
            "note": (
                "Static self-review only: use it to catch obvious mistakes, then still run tests "
                "and manually review scientific/ranking-sensitive changes."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_project_tests(main_window, target: str = "", mode: str = "py_compile", timeout_s: int = 60) -> Dict[str, Any]:
    """Convenience wrapper for common validation commands."""
    mode = str(mode or "py_compile").strip().lower()
    timeout_s = max(1, min(int(timeout_s or 60), 120))
    if mode == "pytest":
        command = ["pytest"]
        if target:
            command.append(target)
    elif mode == "rg":
        return {"error": "Use run_project_command with an explicit rg command for searches."}
    else:
        if target:
            command = ["python", "-m", "py_compile", target]
        else:
            candidates = ["annomass_app.py", "core/agent_tools.py"]
            command = ["python", "-m", "py_compile", *candidates]
    return run_project_command(main_window, command, timeout_s=timeout_s)


def run_project_command(main_window, command: List[str], timeout_s: int = 30) -> Dict[str, Any]:
    """Run a small whitelist of project validation commands without shell."""
    try:
        if not isinstance(command, list) or not command:
            return {"error": "command must be a non-empty array of arguments"}
        args = [str(x) for x in command]
        timeout_s = max(1, min(int(timeout_s or 30), 120))

        allowed = False
        if args[:3] in (["python", "-m", "py_compile"], ["python3", "-m", "py_compile"]):
            allowed = True
        elif args[0] in ("pytest", "python", "python3") and len(args) >= 2:
            if args[0] == "pytest":
                allowed = True
            elif args[1:3] == ["-m", "pytest"]:
                allowed = True
        elif args[:2] == ["git", "status"] or args[:2] == ["git", "diff"]:
            allowed = True
        elif args[0] == "rg":
            allowed = True
        if not allowed:
            return {
                "error": "command is not whitelisted",
                "allowed_examples": [
                    ["python", "-m", "py_compile", "annomass_app.py", "core/agent_tools.py"],
                    ["pytest", "tests/test_example.py"],
                    ["git", "diff", "--", "core/agent_tools.py"],
                    ["rg", "pattern", "core"],
                ],
            }

        for item in args:
            if any(tok in item for tok in (";", "&&", "||", "|", ">", "<", "$(", "`")):
                return {"error": "shell syntax is not allowed; pass plain argv items only"}
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        popen_kwargs = {
            "cwd": str(_PROJECT_ROOT),
            "env": env,
            "text": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = os.setsid
        proc = subprocess.Popen(args, **popen_kwargs)
        if main_window is not None:
            try:
                main_window._agent_active_subprocess = proc
            except Exception:
                pass
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
                stdout, stderr = proc.communicate()
            return {"error": f"command timed out after {timeout_s}s", "stdout": stdout, "stderr": stderr}
        finally:
            if main_window is not None and getattr(main_window, "_agent_active_subprocess", None) is proc:
                try:
                    main_window._agent_active_subprocess = None
                except Exception:
                    pass
        stdout = (stdout or "")[-20000:]
        stderr = (stderr or "")[-20000:]
        return {
            "command": args,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "ok": proc.returncode == 0,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Canonical tool schemas (JSON-schema "parameters", translated per-protocol
# by core.agent_llm_client.translate_tool_schemas)
# ---------------------------------------------------------------------------

def _schema(name, description, properties, required=None):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []},
    }


AGENT_TOOLS: Dict[str, ToolSpec] = {
    "get_pipeline_status": ToolSpec(
        "get_pipeline_status",
        "Check whether spectra are loaded, reference resources are ready, and whether "
        "AnnoMASS has already produced results. Call this before run_pipeline to "
        "self-check preconditions, or before explain_pipeline_results.",
        _schema("get_pipeline_status", "", {})["parameters"],
        "direct", get_pipeline_status,
    ),
    "check_spectrum_quality": ToolSpec(
        "check_spectrum_quality",
        "Rule-based triage of currently loaded spectra (peak count, missing precursor m/z) "
        "BEFORE running AnnoMASS, to flag spectra unlikely to yield a confident "
        "identification. Not a learned quality score -- just objective checks. Useful when "
        "the user just loaded a batch and wants to know if any spectra look unusable.",
        _schema("check_spectrum_quality", "", {
            "min_peaks": {"type": "integer", "description": "Minimum MS2 peak count to not be flagged, default 3"},
        })["parameters"],
        "direct", check_spectrum_quality,
    ),
    "summarize_pipeline_results": ToolSpec(
        "summarize_pipeline_results",
        "Aggregate statistics across the WHOLE current AnnoMASS run (evidence_level / "
        "flavonoid_class / AnnoSM-state distributions, count of same-formula isomer-ambiguous "
        "Top-2 pairs). Use this instead of explain_pipeline_results when the user wants a "
        "batch-level overview rather than one spectrum's candidates.",
        _schema("summarize_pipeline_results", "", {})["parameters"],
        "direct", summarize_pipeline_results,
    ),
    "filter_pipeline_results": ToolSpec(
        "filter_pipeline_results",
        "Preview a filtered subset of the current AnnoMASS table without writing a file. "
        "Use this when the user asks how many rows match a condition, e.g. category A, "
        "rank ranges, classes, or one spectrum. Use export_pipeline_report with the same filters to save.",
        _schema("filter_pipeline_results", "", {
            "filters": {
                "type": "object",
                "description": (
                    "Column filters. Values can be scalar exact match, array exact matches, "
                    "or object with eq/contains/min/max. Examples: "
                    "{\"confidence_level\":\"A\"}, {\"spectrum_rank\":{\"max\":5}}"
                ),
            },
            "limit": {"type": "integer", "description": "Rows to preview, default 20, max 100"},
        })["parameters"],
        "direct", filter_pipeline_results,
    ),
    "export_pipeline_report": ToolSpec(
        "export_pipeline_report",
        "Export the current AnnoMASS results table under the project-local reports/agent_reports directory so the user "
        "has a deliverable artifact (open in Excel, share with a collaborator), not just a "
        "chat explanation. Supports csv, xlsx, and md. Supports deterministic row filters "
        "such as filters={\"confidence_level\":\"A\"}; use filters instead of "
        "asking the user to manually filter CSV files. Returns the file_path.",
        _schema("export_pipeline_report", "", {
            "filename": {"type": "string", "description": "Optional; a timestamped default is used if omitted"},
            "format": {"type": "string", "enum": ["csv", "xlsx", "md"], "description": "Export format; default csv"},
            "filters": {
                "type": "object",
                "description": (
                    "Optional column filters. Values can be a scalar exact match, an array of exact matches, "
                    "or an object with eq/contains/min/max. Examples: "
                    "{\"confidence_level\":\"A\"}, {\"spectrum_rank\":{\"max\":5}}"
                ),
            },
        })["parameters"],
        "direct", export_pipeline_report,
    ),
    "plot_pipeline_results": ToolSpec(
        "plot_pipeline_results",
        "Generate a figure from the current AnnoMASS results and return file_path. "
        "Use only when the user asks to plot/draw/visualize results. Supports PDF "
        "(default), PNG, and SVG outputs; evidence-level distribution, flavonoid-class "
        "distribution, AnnoSM-state distribution, Top-1 score trends by spectrum, "
        "and one spectrum's candidate score comparison.",
        _schema("plot_pipeline_results", "", {
            "plot_type": {
                "type": "string",
                "enum": [
                    "top1_scores",
                    "spectrum_candidates",
                    "evidence_distribution",
                    "class_distribution",
                    "annosm_distribution",
                ],
                "description": "Default top1_scores",
            },
            "filename": {"type": "string", "description": "Optional output filename (.pdf/.png/.svg)"},
            "spectrum_index": {"type": "integer", "description": "Required/used for spectrum_candidates"},
            "format": {"type": "string", "enum": ["pdf", "png", "svg"], "description": "Output format; default pdf"},
        })["parameters"],
        "direct", plot_pipeline_results,
    ),
    "write_pipeline_word_report": ToolSpec(
        "write_pipeline_word_report",
        "Write a Word .docx method/results report from the current AnnoMASS results. "
        "Use this when the user asks for a Word document, method section, results "
        "section, manuscript-style report, or transparent selection-process write-up. "
        "The report includes methods, transparent ranking/selection rules, results "
        "summary, Top-1 per spectrum, and candidate evidence tables.",
        _schema("write_pipeline_word_report", "", {
            "filename": {"type": "string", "description": "Optional .docx filename; a timestamped default is used if omitted"},
            "title": {"type": "string", "description": "Optional report title"},
            "include_top_n": {"type": "integer", "description": "How many ranked candidate rows to include in the evidence table; default 20"},
        })["parameters"],
        "direct", write_pipeline_word_report,
    ),
    "explain_pipeline_results": ToolSpec(
        "explain_pipeline_results",
        "Read the most recent AnnoMASS run's candidate table (does not re-run anything) "
        "and return the top candidates plus the exact ranking-cascade explanation text.",
        _schema("explain_pipeline_results", "", {
            "spectrum_index": {"type": "integer", "description": "Optional: restrict to one spectrum"},
            "top_k": {"type": "integer", "description": "Max candidates to return, default 5"},
        })["parameters"],
        "direct", explain_pipeline_results,
    ),
    "compare_isomers": ToolSpec(
        "compare_isomers",
        "Structured side-by-side comparison of two already-ranked AnnoMASS candidates by their "
        "rank (e.g. Top-1 vs Top-2), for when they look like isomers. Shows each candidate's own "
        "categorized evidence side by side -- does not compute a new ranking score. For "
        "same-formula candidates, also returns diagnostic_fragments: the specific CFIL "
        "characteristic-fragment-ion m/z values and their matched intensity in the query "
        "spectrum that support each side's substitution mode (AnnoSM's own ion-abundance "
        "evidence, broken down from the aggregate annosm_score) -- cite these m/z/intensity "
        "values when explaining why one substitution-mode assignment is better supported.",
        _schema("compare_isomers", "", {
            "rank_a": {"type": "integer"},
            "rank_b": {"type": "integer"},
            "spectrum_index": {"type": "integer", "description": "Optional: restrict to one spectrum"},
        }, required=["rank_a", "rank_b"])["parameters"],
        "direct", compare_isomers,
    ),
    "query_structure_database": ToolSpec(
        "query_structure_database",
        "Search the live flavonoid structure database (name/InChIKey/formula/SMILES) "
        "for a compound's structural record.",
        _schema("query_structure_database", "", {
            "query": {"type": "string", "description": "Name, InChIKey, formula, or SMILES"},
            "field": {"type": "string", "enum": ["auto", "name", "inchikey", "formula", "smiles"]},
            "limit": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", query_structure_database,
    ),
    "query_reference_spectra": ToolSpec(
        "query_reference_spectra",
        "Check whether a compound has reference MS/MS spectra in the project's 4 "
        "canonical reference datasets (MassBank standard-reference, MassSpecGym, GNPS, MoNA) "
        "and from which source(s).",
        _schema("query_reference_spectra", "", {
            "query": {"type": "string", "description": "Name, InChIKey, formula, or SMILES"},
            "field": {"type": "string", "enum": ["auto", "name", "inchikey", "formula", "smiles"]},
            "source": {"type": "string", "description": "'all' or one of MassBank/MassSpecGym/GNPS/MoNA"},
            "limit": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", query_reference_spectra,
    ),
    "run_abcort_predict": ToolSpec(
        "run_abcort_predict",
        "Predict retention time (minutes) for a list of SMILES using the ABCoRT GNN model.",
        _schema("run_abcort_predict", "", {
            "smiles_list": {"type": "array", "items": {"type": "string"}},
            "checkpoint": {"type": "string", "description": "Optional checkpoint path; default is the pretrained SMRT model"},
        }, required=["smiles_list"])["parameters"],
        "direct", run_abcort_predict,
    ),
    "remember_fact": ToolSpec(
        "remember_fact",
        "Save a fact to persistent memory, loaded back in on every future conversation. Use for "
        "durable preferences/project facts, not one-off task details (see system prompt).",
        _schema("remember_fact", "", {
            "fact": {"type": "string"},
            "category": {"type": "string", "description": "Optional free-text tag, e.g. 'preference', 'project'"},
        }, required=["fact"])["parameters"],
        "direct", remember_fact,
    ),
    "list_agent_skills": ToolSpec(
        "list_agent_skills",
        "List installed local Markdown skills loaded into the Agent prompt. Use when the "
        "user asks what skills/capabilities are installed.",
        _schema("list_agent_skills", "", {})["parameters"],
        "direct", list_agent_skills,
    ),
    "install_agent_skill": ToolSpec(
        "install_agent_skill",
        "Install or update a reusable local Agent skill from Markdown content or a local "
        "Markdown file path. A skill is loaded into future Agent turns as task guidance. "
        "Use only when the user explicitly asks to install/update a skill.",
        _schema("install_agent_skill", "", {
            "name": {"type": "string", "description": "Skill name, e.g. method_results_writer"},
            "content": {"type": "string", "description": "Markdown skill instructions; optional if source_path is given"},
            "source_path": {"type": "string", "description": "Local Markdown file to install; optional if content is given"},
        }, required=["name"])["parameters"],
        "direct", install_agent_skill,
    ),
    "delete_agent_skill": ToolSpec(
        "delete_agent_skill",
        "Delete an installed local Agent skill by name. Use only when the user explicitly "
        "asks to remove/delete a skill.",
        _schema("delete_agent_skill", "", {
            "name": {"type": "string", "description": "Installed skill name"},
        }, required=["name"])["parameters"],
        "direct", delete_agent_skill,
    ),
    "read_project_file": ToolSpec(
        "read_project_file",
        "Read a text/code file inside the project workspace. Use before editing a file, "
        "or when the user asks how the GUI/code currently works. Protected areas such as "
        ".git, model, data, virtualenvs, and caches are blocked.",
        _schema("read_project_file", "", {
            "path": {"type": "string", "description": "Project-relative or absolute path inside the workspace"},
            "max_chars": {"type": "integer", "description": "Maximum characters to return; default 30000"},
        }, required=["path"])["parameters"],
        "direct", read_project_file,
    ),
    "list_project_files": ToolSpec(
        "list_project_files",
        "List project files under a workspace path while skipping protected/heavy directories "
        "such as .git, model, data, virtualenvs, and caches. Use to inspect project structure "
        "before searching or editing.",
        _schema("list_project_files", "", {
            "root": {"type": "string", "description": "Project-relative root path; default ."},
            "file_glob": {"type": "string", "description": "Optional glob such as '*.py' or 'core/*.py'"},
            "max_files": {"type": "integer", "description": "Maximum files to return; default 200"},
        })["parameters"],
        "direct", list_project_files,
    ),
    "search_project": ToolSpec(
        "search_project",
        "Search project text/code files with ripgrep (or a Python fallback). Use to locate "
        "functions, settings, UI strings, or code paths before editing.",
        _schema("search_project", "", {
            "query": {"type": "string", "description": "Text or regex to search for"},
            "file_glob": {"type": "string", "description": "Optional rg-style glob, e.g. '*.py' or 'core/*.py'"},
            "max_matches": {"type": "integer", "description": "Maximum matches to return; default 80"},
            "literal": {"type": "boolean", "description": "Use fixed-string search; default true"},
        }, required=["query"])["parameters"],
        "direct", search_project,
    ),
    "write_project_file": ToolSpec(
        "write_project_file",
        "Write a permitted text/code file inside the project workspace. Use only after "
        "reading the relevant file and only for explicit code-edit/debug requests. "
        "Pass expected_current_content when replacing an existing file to avoid "
        "overwriting concurrent user edits. After writing, call get_project_diff and "
        "run_project_command for validation.",
        _schema("write_project_file", "", {
            "path": {"type": "string", "description": "Project-relative or absolute path inside the workspace"},
            "content": {"type": "string", "description": "Complete new file content"},
            "expected_current_content": {"type": "string", "description": "Optional exact current content read earlier"},
            "create": {"type": "boolean", "description": "Set true only when creating a new file"},
        }, required=["path", "content"])["parameters"],
        "direct", write_project_file,
    ),
    "apply_project_patch": ToolSpec(
        "apply_project_patch",
        "Apply a unified diff patch to one or more permitted text/code files in the project. "
        "Use for Codex-style surgical edits after reading/searching the relevant code. "
        "The tool validates all touched paths, blocks protected directories and file "
        "deletions, runs git apply --check first, and returns touched_files. After applying, "
        "call get_project_diff and run_project_tests or an allowed validation command.",
        _schema("apply_project_patch", "", {
            "patch": {"type": "string", "description": "Unified diff text"},
            "check_only": {"type": "boolean", "description": "If true, validate without applying"},
        }, required=["patch"])["parameters"],
        "direct", apply_project_patch,
    ),
    "get_project_diff": ToolSpec(
        "get_project_diff",
        "Return git diff for the project or one file after code edits. Use this before "
        "summarizing any code change; use it to describe what changed without dumping "
        "large diffs unless the user asks.",
        _schema("get_project_diff", "", {
            "path": {"type": "string", "description": "Optional project-relative file path"},
            "max_chars": {"type": "integer", "description": "Maximum diff characters to return; default 40000"},
        })["parameters"],
        "direct", get_project_diff,
    ),
    "summarize_project_changes": ToolSpec(
        "summarize_project_changes",
        "Return changed file names plus the current git diff for Codex-style final "
        "summaries after code edits. Use near the end of a code-modification task.",
        _schema("summarize_project_changes", "", {
            "max_chars": {"type": "integer", "description": "Maximum diff characters to return; default 40000"},
        })["parameters"],
        "direct", summarize_project_changes,
    ),
    "review_project_code": ToolSpec(
        "review_project_code",
        "Run a lightweight code self-review over changed files or a target path. Checks "
        "Python syntax, unresolved merge markers, risky command/eval patterns, broad "
        "exception handling, and ranking/scientific-evidence-sensitive code. Use when "
        "the user asks for code self-check/review/自查, and after code edits before "
        "summarizing the change.",
        _schema("review_project_code", "", {
            "target": {"type": "string", "description": "Optional file or directory to review; omit for changed files"},
            "scope": {"type": "string", "enum": ["changed", "all"], "description": "Review changed files by default; all scans the project"},
            "max_files": {"type": "integer", "description": "Maximum files to review; default 80"},
            "max_findings": {"type": "integer", "description": "Maximum findings to return; default 80"},
        })["parameters"],
        "direct", review_project_code,
    ),
    "run_project_tests": ToolSpec(
        "run_project_tests",
        "Convenience validation wrapper after edits. mode=py_compile compiles the target "
        "file or core GUI files by default; mode=pytest runs pytest optionally scoped to "
        "target. This uses the same command whitelist as run_project_command. Include "
        "the returned pass/fail status in the final change summary.",
        _schema("run_project_tests", "", {
            "target": {"type": "string", "description": "Optional file/test target"},
            "mode": {"type": "string", "enum": ["py_compile", "pytest"], "description": "Validation mode; default py_compile"},
            "timeout_s": {"type": "integer", "description": "Timeout seconds, capped at 120"},
        })["parameters"],
        "direct", run_project_tests,
    ),
    "run_project_command": ToolSpec(
        "run_project_command",
        "Run a whitelisted project validation command without shell. Allowed examples: "
        "python -m py_compile ..., pytest ..., git status/diff, and rg. Use after code "
        "edits or when the user asks for a validation run. Include the returned "
        "command and pass/fail status in the final change summary.",
        _schema("run_project_command", "", {
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command argv list, not a shell string",
            },
            "timeout_s": {"type": "integer", "description": "Timeout seconds, capped at 120"},
        }, required=["command"])["parameters"],
        "direct", run_project_command,
    ),
    "query_massql": ToolSpec(
        "query_massql",
        "Run a MassQL-style query against spectra currently loaded in the app. Use this when "
        "the user sends a MassQL command without specifying a separate file or URL. Useful "
        "for diagnostic fragment ions and neutral losses. Supports MS2PROD, MS2NL, "
        "MS1MZ/PRECURSORMZ, RT, TOLERANCEMZ, TOLERANCEPPM, MININT, and LIMIT in the local subset.",
        _schema("query_massql", "", {
            "query": {
                "type": "string",
                "description": "Example: WHERE MS2NL = 162.0528:TOLERANCEMZ=0.02 LIMIT 100",
            },
            "limit": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", query_massql,
    ),
    "query_massql_file": ToolSpec(
        "query_massql_file",
        "Run the official MassQL engine on a local MS data/library file, such as an MGF, mzML, "
        "mzXML, JSON, TXT, or DAT file. Use this for local spectral libraries or local GNPS/"
        "MassIVE downloads when the user provides a file path.",
        _schema("query_massql_file", "", {
            "query": {
                "type": "string",
                "description": "MassQL query, e.g. QUERY scaninfo(MS2DATA) WHERE MS2PROD=271.0601:TOLERANCEMZ=0.01",
            },
            "file_path": {"type": "string", "description": "Local MGF/mzML/mzXML/json/txt/dat file path."},
            "limit": {"type": "integer"},
        }, required=["query", "file_path"])["parameters"],
        "direct", query_massql_file,
    ),
    "query_massql_gnps": ToolSpec(
        "query_massql_gnps",
        "Use the public GNPS2 MassQL online service. This sends the MassQL query to the "
        "GNPS2 MassQL Sandbox to parse/explain it and generate a ProteoSAFe MSQL-NF remote "
        "workflow link. Use when the user asks for online/GNPS/public-library MassQL search.",
        _schema("query_massql_gnps", "", {
            "query": {"type": "string"},
        }, required=["query"])["parameters"],
        "direct", query_massql_gnps,
    ),
    "query_massql_libraries": ToolSpec(
        "query_massql_libraries",
        "Prepare online MassQL routes across multiple spectral-library/public-data resources "
        "(GNPS/MassIVE ProteoSAFe workflow, GNPS2 MassQL Library Searching, MassQL Compendium, "
        "and MassQL Sandbox). Use this when the user asks to use MassQL across multiple online "
        "libraries for spectrum interpretation. Returns actionable links and query explanations; "
        "some resources require interactive workflow submission rather than synchronous API results.",
        _schema("query_massql_libraries", "", {
            "query": {"type": "string"},
            "libraries": {
                "type": "array",
                "items": {"type": "string", "enum": ["gnps", "massive", "libraries", "compendium"]},
                "description": "Optional online resources to route to; omit for all.",
            },
        }, required=["query"])["parameters"],
        "direct", query_massql_libraries,
    ),
    "query_massql_remote_file": ToolSpec(
        "query_massql_remote_file",
        "Download a direct remote MS data URL and run the official MassQL engine locally. "
        "Use only when the user provides a direct downloadable MGF/mzML/mzXML file URL; for "
        "online GNPS workflow/library search use query_massql_gnps instead.",
        _schema("query_massql_remote_file", "", {
            "query": {"type": "string"},
            "source_url": {"type": "string", "description": "Direct http(s) URL to MGF/mzML/mzXML/json/txt/dat."},
            "limit": {"type": "integer"},
            "max_mb": {"type": "integer"},
        }, required=["query", "source_url"])["parameters"],
        "direct", query_massql_remote_file,
    ),
    "query_pubchem": ToolSpec(
        "query_pubchem",
        "Look up a compound on PubChem (live internet query) by name, SMILES, or InChIKey -- use "
        "when query_structure_database finds nothing locally. Returns formula/SMILES/IUPAC name/"
        "molecular weight/InChIKey/PubChem CID.",
        _schema("query_pubchem", "", {
            "query": {"type": "string"},
            "field": {"type": "string", "enum": ["auto", "name", "smiles", "inchikey"]},
            "limit": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", query_pubchem,
    ),
    "search_literature": ToolSpec(
        "search_literature",
        "Search PubMed (live internet query) for papers relevant to a compound, spectrum, or "
        "analytical method, to help explain or support a result with citations.",
        _schema("search_literature", "", {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", search_literature,
    ),
    "query_massbank_online": ToolSpec(
        "query_massbank_online",
        "Look up real MS/MS records on the public MassBank database (live internet query, via "
        "massbank.eu) by compound name or InChIKey -- covers compounds beyond the project's local "
        "154-compound curated MassBank standard-reference subset. Returns per-record summaries "
        "(accession, formula, SMILES, InChIKey, ion mode, precursor, instrument, retention time, "
        "peak count) -- NOT full peak lists, and a broad name query can match hundreds of records "
        "(see total_matches), so prefer InChIKey when you have one.",
        _schema("query_massbank_online", "", {
            "query": {"type": "string"},
            "field": {"type": "string", "enum": ["auto", "name", "inchikey"]},
            "limit": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", query_massbank_online,
    ),
    "query_mona_online": ToolSpec(
        "query_mona_online",
        "Look up records in MoNA / MassBank of North America (live internet query) by compound "
        "name, InChIKey, SPLASH, precursor m/z, or raw MoNA RSQL. Returns spectrum-record "
        "summaries when MoNA's filtered REST API is available; if MoNA requires authorization, "
        "returns MoNA browser/download links and the exact RSQL query. MoNA is an external "
        "spectral library; for MassQL-style querying, download a MoNA export and use "
        "query_massql_file on the local file.",
        _schema("query_mona_online", "", {
            "query": {"type": "string"},
            "field": {"type": "string", "enum": ["auto", "name", "inchikey", "splash", "precursor_mz", "rsql"]},
            "limit": {"type": "integer"},
        }, required=["query"])["parameters"],
        "direct", query_mona_online,
    ),
    "load_spectra_file": ToolSpec(
        "load_spectra_file",
        "Load one or more MGF/MAT/MSP spectra files by path into the app (equivalent to using the "
        "Open button with multi-select), so run_pipeline has data to work with. Prefer file_paths "
        "when the user provides a batch of sample files; do not ask the user to manually Open them.",
        _schema("load_spectra_file", "", {
            "file_path": {"type": "string", "description": "Single MGF/MAT/MSP path; kept for backward compatibility"},
            "file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple MGF/MAT/MSP paths to load together",
            },
        })["parameters"],
        "bridge", None,  # implemented in annomass_app.py: AgentWorker._tool_load_spectra_file
    ),
    "run_pipeline": ToolSpec(
        "run_pipeline",
        "Run the full AnnoMASS identification algorithm (structure retrieval -> spectral-library search -> Spectrum Predictor -> rank) "
        "using whatever spectra are currently loaded and whatever parameters are currently "
        "already set in the app. Takes no parameters -- it cannot change ranking-relevant "
        "settings. Call get_pipeline_status first if unsure whether preconditions are met.",
        _schema("run_pipeline", "", {})["parameters"],
        "bridge", None,  # implemented in annomass_app.py: AgentWorker._tool_run_pipeline
    ),
    "show_spectrum_plot": ToolSpec(
        "show_spectrum_plot",
        "Display a loaded+identified spectrum's plot in the Spectrum View / Structure View "
        "panel (same as manually clicking that spectrum in the list). Purely for showing the "
        "user a chart; does not change any ranking or data.",
        _schema("show_spectrum_plot", "", {
            "spectrum_index": {"type": "integer", "description": "Index of the spectrum (as used in AnnoMASS results/spectrum_index column)"},
        }, required=["spectrum_index"])["parameters"],
        "bridge", None,  # implemented in annomass_app.py: AgentWorker._tool_show_spectrum_plot
    ),
    "show_structure_plot": ToolSpec(
        "show_structure_plot",
        "Render an arbitrary compound's 2D structure (by SMILES) into the Structure View "
        "panel, independent of the currently selected spectrum/candidate. Pass compound_name "
        "when it is known (e.g. from the AnnoMASS results table) so the exported/saved image "
        "file is named after the compound instead of a random string.",
        _schema("show_structure_plot", "", {
            "smiles": {"type": "string"},
            "compound_name": {"type": "string", "description": "Compound name to use for the exported file name, if known"},
        }, required=["smiles"])["parameters"],
        "bridge", None,  # implemented in annomass_app.py: AgentWorker._tool_show_structure_plot
    ),
    "open_file": ToolSpec(
        "open_file",
        "Open a file in the app's built-in file viewer window (so the user doesn't have to switch "
        "to another program to see a file you reference or generate). Supports PDF, images "
        "(PNG/JPG/BMP/GIF/TIFF/SVG), code/text/spectra formats (PY/TXT/LOG/MD/MGF/MSP/MAT/JSON), "
        "CSV/TSV and Excel workbooks as spreadsheet-like tables, and Word documents "
        "(.docx -- legacy .doc is not supported, only .docx).",
        _schema("open_file", "", {
            "file_path": {"type": "string"},
        }, required=["file_path"])["parameters"],
        "bridge", None,  # implemented in annomass_app.py: AgentWorker._tool_open_file
    ),
    "save_current_preview": ToolSpec(
        "save_current_preview",
        "Save the file/image currently visible in the Agent preview panel to an exact path and file name. "
        "Use this when the user asks conversationally to save, copy, rename, or export the currently shown "
        "spectrum/structure/image. The destination_path must include the desired directory and file name. "
        "For raster images, the destination extension can convert the format (PNG/JPG/TIFF/BMP). Never set "
        "overwrite=true unless the user explicitly asks to replace an existing file.",
        _schema("save_current_preview", "", {
            "destination_path": {
                "type": "string",
                "description": "Exact destination path including directory, file name, and preferably extension",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Replace an existing destination only when the user explicitly requested overwrite; default false",
            },
        }, required=["destination_path"])["parameters"],
        "bridge", None,
    ),
}


def canonical_tool_schemas(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    keys = names or list(AGENT_TOOLS.keys())
    return [
        {"name": AGENT_TOOLS[k].name, "description": AGENT_TOOLS[k].description, "parameters": AGENT_TOOLS[k].parameters}
        for k in keys
    ]


# ---------------------------------------------------------------------------
# Pure tool-calling loop -- no Qt import, unit-testable with a stub client.
# ---------------------------------------------------------------------------

# Context-window management: trigger compaction once real reported usage
# crosses this fraction of the provider's (approximate) context_window --
# see core/agent_config.py's PROVIDER_PRESETS "context_window" comment for
# why it's a best-effort budget, not an exact guarantee.
COMPACT_THRESHOLD_RATIO = 0.75
KEEP_RECENT_MESSAGES = 4
DEFAULT_TOOL_ITERATIONS = 16
EXTENDED_TOOL_ITERATIONS = 24
MAX_DYNAMIC_TOOL_ITERATIONS = 32
MAX_CONSECUTIVE_TOOL_ERRORS = 5


def sanitize_persistent_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a provider-safe long-term chat history.

    Tool-call messages are only valid inside the same request/response loop:
    an OpenAI-compatible provider expects every ``role=tool`` message to be
    immediately tied to a preceding assistant message with matching
    ``tool_calls``. Persisting those intermediate messages across GUI turns is
    brittle and DeepSeek rejects orphaned histories with HTTP 400. The visible
    tool timeline stays in session["tool_log"]/display_log; the model history
    keeps only natural-language user/assistant turns.
    """
    clean: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            continue
        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content:
                clean.append({"role": "assistant", "content": content})
            continue
        if role == "assistant":
            content = msg.get("content", "")
            # Anthropic tool-use blocks live inside assistant content; they
            # have the same one-turn validity as OpenAI tool_calls.
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(t for t in text_parts if t)
            clean.append({"role": "assistant", "content": str(content or "")})
            continue
        if role == "user":
            content = msg.get("content", "")
            # Anthropic tool_result blocks are stored as user content; do not
            # persist them into later turns.
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if not text_parts:
                    continue
                content = "\n".join(text_parts)
            clean.append({"role": "user", "content": str(content or "")})
    return clean


def _compact_messages(client, messages, system_prompt, keep_recent=KEEP_RECENT_MESSAGES):
    """Summarize everything except the last `keep_recent` messages into one
    synthetic message, via an extra (tools-disabled) call to the same model.
    Automatic context compaction. Mutates nothing in place --
    caller assigns the result back (messages[:] = ...) to keep the shared
    list object identity, since AgentWorker/session["messages"] is passed
    by reference."""
    if len(messages) <= keep_recent:
        return messages
    older, recent = messages[:-keep_recent], messages[-keep_recent:]

    transcript_lines = []
    for m in older:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False, default=str)
        transcript_lines.append(f"{role}: {content}")

    summary_request = [{
        "role": "user",
        "content": (
            "Summarize the following conversation transcript concisely, "
            "preserving any concrete facts, decisions, or results a "
            "continuing conversation would need. Plain text, no preamble:\n\n"
            + "\n".join(transcript_lines)
        ),
    }]
    try:
        summary_reply = client.chat(summary_request, tools=None, system=system_prompt)
        summary_text = summary_reply.text or "(summary unavailable)"
    except Exception as exc:
        summary_text = f"(compaction summary failed: {exc}; earlier turns dropped)"

    summary_message = {"role": "user", "content": "[Earlier conversation summary]\n" + summary_text}
    return [summary_message] + recent


_FILE_PRODUCING_TOOLS = {
    "export_pipeline_report",
    "plot_pipeline_results",
    "write_pipeline_word_report",
    "write_project_file",
    "apply_project_patch",
    "save_current_preview",
}


def _tool_truth_audit(records: List[Dict[str, Any]]) -> str:
    """Build a compact facts-only audit for the final answer.

    This is deliberately deterministic guardrail text, not a replacement for
    model judgment. It prevents a failed write/open/validation result from
    becoming a confident prose claim such as "the file exists".
    """
    if not records:
        return ""
    problems = []
    verified_outputs = []
    for rec in records:
        name = str(rec.get("name") or "")
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        args = rec.get("args") if isinstance(rec.get("args"), dict) else {}
        if rec.get("status") == "failed" or result.get("error"):
            problems.append(f"- {name} failed: {result.get('error') or 'unknown error'}")
            continue
        if name in _FILE_PRODUCING_TOOLS:
            if result.get("verified") is True:
                file_path = result.get("file_path") or result.get("path") or ", ".join(result.get("changed_files") or [])
                verified_outputs.append(f"- {name} verified output: {file_path}")
            else:
                target = result.get("file_path") or result.get("path") or args.get("path") or args.get("filename") or "(unknown target)"
                problems.append(f"- {name} did not return verified=true for {target}; do not claim it succeeded.")
    if not problems:
        return ""
    lines = [
        "[Tool result audit before final answer]",
        "You must base the final answer only on these tool results.",
        "Do not say a file was created, opened, moved, validated, or exists unless the relevant tool returned success and verified=true when applicable.",
        "If a tool failed or was not verified, state that plainly and give the exact failure/uncertainty.",
        "Problems:",
        *problems,
    ]
    if verified_outputs:
        lines.extend(["Verified outputs:", *verified_outputs])
    return "\n".join(lines)


def _final_reply_with_truth_audit(client, messages, draft_reply, records, system_prompt, report_usage):
    audit = _tool_truth_audit(records)
    if not audit:
        return draft_reply.text or ""
    temp_messages = list(messages)
    try:
        temp_messages.append(client.build_assistant_message(draft_reply))
    except Exception:
        if draft_reply.text:
            temp_messages.append({"role": "assistant", "content": draft_reply.text})
    temp_messages.append({
        "role": "user",
        "content": audit + "\n\nRewrite your final answer now, correcting any overclaim.",
    })
    final_reply = client.chat(temp_messages, tools=None, system=system_prompt)
    report_usage(final_reply)
    return final_reply.text or ""


def run_agent_loop(client, tool_schemas_translated, messages, dispatch_fn,
                    system_prompt=SYSTEM_PROMPT, progress_cb=None, max_iters=DEFAULT_TOOL_ITERATIONS,
                    context_window=None, tool_log=None, should_stop=None,
                    usage_cb=None):
    """messages: mutable list of provider-native message dicts (already
    contains the new user turn). dispatch_fn(name, args) -> dict result.
    Returns the final assistant text (possibly with a truncation notice).
    context_window: optional approximate token budget (see
    core/agent_config.py) -- when the provider's own reported usage crosses
    COMPACT_THRESHOLD_RATIO of it, older messages get auto-summarized.
    tool_log: optional mutable list (like messages) that each call appends
    {"id","name","args","result","duration_s","status"} to -- the "Agent
    execution timeline" detail view (annomass_app.py's
    _agent_chat_link_clicked) reads this to show a tool call's full
    input/output on click, since the chat log itself only ever shows a
    one-line "✓ name" summary."""
    def stop_requested():
        try:
            return bool(should_stop and should_stop())
        except Exception:
            return False

    def report_usage(reply):
        if not usage_cb or not getattr(reply, "usage", None):
            return
        usage = dict(reply.usage or {})
        total = int(usage.get("total_tokens") or 0)
        window = int(context_window or 0)
        payload = {
            "usage": usage,
            "context_window": window,
            "ratio": (float(total) / float(window)) if window > 0 and total >= 0 else None,
            "compact_threshold_ratio": COMPACT_THRESHOLD_RATIO,
        }
        try:
            usage_cb(payload)
        except Exception:
            pass

    budget = max(1, int(max_iters or DEFAULT_TOOL_ITERATIONS))
    budget = min(max(budget, DEFAULT_TOOL_ITERATIONS), MAX_DYNAMIC_TOOL_ITERATIONS)
    consecutive_errors = 0
    completed_calls = 0
    run_records: List[Dict[str, Any]] = []

    i = 0
    while i < budget:
        if stop_requested():
            return "[stopped by user]"
        if progress_cb:
            progress_cb("Thinking..." if i == 0 else f"Thinking (step {i + 1}/{budget})...")
        reply = client.chat(messages, tools=tool_schemas_translated, system=system_prompt)
        report_usage(reply)
        if stop_requested():
            return "[stopped by user]"

        if context_window and reply.usage and reply.usage.get("total_tokens", 0) > context_window * COMPACT_THRESHOLD_RATIO:
            if progress_cb:
                progress_cb("Compacting conversation history...")
            messages[:] = _compact_messages(client, messages, system_prompt)

        if not reply.tool_calls:
            return _final_reply_with_truth_audit(client, messages, reply, run_records, system_prompt, report_usage)

        messages.append(client.build_assistant_message(reply))
        for tool_call in reply.tool_calls:
            if progress_cb:
                progress_cb(f"Calling tool: {tool_call.name}...")
            # monotonic(), not time() -- a wall-clock adjustment (e.g. NTP
            # sync) mid-call would otherwise produce a negative duration.
            start = time.monotonic()
            try:
                result = dispatch_fn(tool_call.name, tool_call.arguments)
                duration = time.monotonic() - start
                status = "failed" if isinstance(result, dict) and result.get("error") else "completed"
                # Two progress events per call: the "Calling tool:" one above
                # is transient (status label only, see _agent_on_progress),
                # this one is the persistent checklist-style entry the chat
                # log keeps -- "✓" so a finished call reads as a checked
                # item rather than a repeated "in progress" line.
                mark = "✓" if status == "completed" else "✗"
                done_msg = f"{mark} {tool_call.name} ({duration:.1f}s)" + (f" -- error: {result['error']}" if status == "failed" else "")
            except Exception as exc:
                duration = time.monotonic() - start
                result = {"error": str(exc)}
                status = "failed"
                done_msg = f"✗ {tool_call.name} ({duration:.1f}s) -- error: {exc}"
            # Append to tool_log BEFORE calling progress_cb -- the UI (see
            # _agent_on_progress/_agent_chat_link_clicked in annomass_app.py)
            # correlates the "✓ name (Ns)" message with tool_log[-1], so the
            # record must already exist by the time that callback runs.
            if tool_log is not None:
                record = {
                    "id": len(tool_log),
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                    "result": result,
                    "duration_s": round(duration, 2),
                    "status": status,
                }
                tool_log.append(record)
            else:
                record = {
                    "id": len(run_records),
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                    "result": result,
                    "duration_s": round(duration, 2),
                    "status": status,
                }
            run_records.append(record)
            if progress_cb:
                progress_cb(done_msg)
            messages.append(client.build_tool_result_message(tool_call, result))
            if status == "failed":
                consecutive_errors += 1
            else:
                consecutive_errors = 0
                completed_calls += 1
            if consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                if progress_cb:
                    progress_cb("Repeated tool failures; requesting final answer...")
                final_reply = client.chat(messages, tools=None, system=system_prompt)
                report_usage(final_reply)
                text = _final_reply_with_truth_audit(client, messages, final_reply, run_records, system_prompt, report_usage)
                return text + "\n\n[stopped: repeated tool failures]"
            if (
                i + 1 >= budget
                and budget < MAX_DYNAMIC_TOOL_ITERATIONS
                and completed_calls >= max(2, budget // 2)
                and consecutive_errors == 0
            ):
                new_budget = min(MAX_DYNAMIC_TOOL_ITERATIONS, EXTENDED_TOOL_ITERATIONS if budget < EXTENDED_TOOL_ITERATIONS else budget + 4)
                if new_budget > budget:
                    budget = new_budget
                    if progress_cb:
                        progress_cb(f"Extending tool budget to {budget} steps...")
            if stop_requested():
                return "[stopped by user]"
        i += 1

    # Ran out of iterations while the model still wanted to call tools --
    # ask once more with tools disabled to get a final answer instead of
    # silently dropping whatever partial progress was made.
    if progress_cb:
        progress_cb("Tool-call limit reached, requesting final answer...")
    if stop_requested():
        return "[stopped by user]"
    final_reply = client.chat(messages, tools=None, system=system_prompt)
    report_usage(final_reply)
    text = _final_reply_with_truth_audit(client, messages, final_reply, run_records, system_prompt, report_usage)
    return text + "\n\n[stopped: tool-call limit reached]"
