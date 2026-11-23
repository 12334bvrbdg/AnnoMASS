# core/: the AnnoMASS pipeline

| File | What it does |
|---|---|
| `annomass_algorithm.py` | Main pipeline used by both the GUI and the command line: reads the spectra, runs levels A → B → C (each spectrum stops at the first level that passes), then level D, then the substituent-position check and Top-K output |
| `staged_levels.py` | Scores, weights and thresholds of levels A–D |
| `paths.py` | Where AnnoMASS finds the third-party code and environments (`ANNOMASS_HOME`, default `~/AnnoMASS`) |
| `dreams_adapter.py`, `_dreams_search_worker.py` | Level A: spectral-embedding search of the reference library (runs in the `spectrum-retrieval` environment) |
| `annotating/` | Levels B/C: candidate structures from the structure library, formula and structure scoring |
| `icefrag_adapter.py` | Levels C/D: predicted spectra of candidate structures (runs `scripts/icefrag/run_iceberg.py` in the `spectrum-prediction` environment) |
| `deepmet_clm_adapter.py` | Level D: generated flavonoid structures matched by precursor mass |
| `annosm_bridge.py` | Substituent-position check with the diagnostic fragment libraries (CFIL/TPIL) |
| `abcort_bridge.py` | Optional retention-time prediction |
| `flavclass_filter.py` | Flavonoid subclass recognition (also used by `flavonoid_screening/`) |
| `flavonoid_fragment_scorer.py` | Flavonoid diagnostic-fragment scoring |
| `importing/` | Reading MGF and MSP files |
| `Spectrum.py`, `main.py`, `pycdk.py` | Spectrum container, worker threads and CDK helpers |
| `result_table_export.py`, `result_figure_exporter.py` | Excel result table and figure export |
| `structure_db_query.py`, `pubchem_lookup.py`, `mona_lookup.py`, `massbank_lookup.py`, `pubmed_search.py`, `massql.py` | Lookups used by the GUI and its assistant panel |
| `agent_*.py`, `i18n.py`, `file_viewer_widget.py` | Assistant panel of the GUI (optional; needs your own LLM API key) |
| `fiddle_adapter.py`, `flavonoid_reference_db.py` | Optional helpers; AnnoMASS runs without them |

Part of `Spectrum.py`, `main.py`, `pycdk.py`, `annotating/` and `importing/` is derived from
[DeepMASS2](https://github.com/hcji/DeepMASS2_GUI) (AGPL-3.0).
