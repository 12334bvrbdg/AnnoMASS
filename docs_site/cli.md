---
description: Run AnnoMASS from the command line for batch annotation of flavonoid MS/MS spectra. Commands, options, outputs and example scripts.
---

# Command line (CLI)

The command line runs the same pipeline as the GUI, without a window. Use it for large
batches, for servers without a screen, or in scripts.

## Basic use

```bash
annomass headless --mgf input.mgf --out results.xlsx
```

- `--mgf`: input file (MGF)
- `--out`: output table, `.xlsx` or `.csv`

This runs all levels (A–D) with the default settings and keeps up to 10 candidates per spectrum.

!!! note "Without the launcher"
    `annomass headless ...` is a short form of the following (useful on servers, or when the
    launcher is not on your `PATH`):

    ```bash
    cd /path/to/AnnoMASS
    ~/AnnoMASS/envs/main/bin/python run_pipeline_headless.py --mgf input.mgf --out results.xlsx
    ```

## Common examples

```bash
# Top 3 candidates per spectrum (same as the GUI)
annomass headless --mgf input.mgf --out results.xlsx --top-n 3

# Library matching only (fast)
annomass headless --mgf input.mgf --out results.xlsx --modules A

# Levels A–C only, skip generated structures (faster)
annomass headless --mgf input.mgf --out results.xlsx --modules ABC

# Your instrument: Orbitrap, normalised collision energy 35
annomass headless --mgf input.mgf --out results.xlsx \
    --iceberg-instrument Orbitrap --iceberg-nce 35

# Table only, no figures (faster, less disk space)
annomass headless --mgf input.mgf --out results.csv --skip-figures
```

## Options

### Input and output

| Option | Default | Meaning |
|---|---|---|
| `--mgf FILE` | | Input MGF file |
| `--out FILE` | `pipeline_headless_results.xlsx` | Output table. The extension (`.xlsx` or `.csv`) sets the format. |
| `--top-n N` | 10 | Candidates kept per spectrum |
| `--skip-figures` | off | Do not save spectrum comparison plots and fragment figures |
| `--per-spectrum-dedup` | off | Report each spectrum separately. By default, spectra that give the same compound are merged into one row. |

### Levels

| Option | Default | Meaning |
|---|---|---|
| `--modules LEVELS` | `ABCD` | Levels to run, in any combination: `A`, `BC`, `ABC`, `A,C` ... |
| `--disable-deepmet-clm` | | Do not run level D (same as leaving `D` out of `--modules`) |

### Spectrum quality and mass tolerance

| Option | Default | Meaning |
|---|---|---|
| `--ppm X` | 10 | Precursor mass tolerance in ppm. AnnoMASS adds a small extra tolerance when the precursor *m/z* is written with few decimals. |
| `--min-fragment-peaks N` | 3 | Skip spectra with fewer fragment peaks (≥1% of the base peak, more than 2 Da from the precursor). `0` keeps all spectra. |

### Predicted spectra (levels C and D)

| Option | Default | Meaning |
|---|---|---|
| `--iceberg-nce X` | 30 | Collision energy used for the predicted spectra |
| `--iceberg-instrument NAME` | `Unknown` | `Orbitrap`, `QTOF`, `IT-FT` or `Unknown` |
| `--iceberg-peak-min-rel X` | 0.01 | Peaks below this fraction of the base peak are removed before comparison |
| `--iceberg-peak-top-k N` | 80 | Keep at most N strongest peaks (0 = all) |

### Optional refinements

| Option | Default | Meaning |
|---|---|---|
| `--enable-abcort-rt` | off | Add a predicted retention time column (for information only) |
| `--enable-fiddle` | off | Remove candidates whose formula does not fit the spectrum |
| `--disable-annosm-rerank` | | Turn off the substituent-position check |

To see every option, including those for method development:

```bash
annomass headless --help
```

!!! warning "Weights and thresholds"
    Options such as `--level-a-weight` or `--level-b-threshold` exist for method development.
    Use the default values for normal analyses, so that your results can be compared with
    published ones.

## Output

| Output | Content |
|---|---|
| `results.xlsx` (or `.csv`) | The result table. See [Input and output](input-output.md#result-table). |
| `final_results/figures/spectra_comparison/` | For each candidate: your spectrum against the reference or predicted spectrum |
| `final_results/figures/iceberg_fragments/` | Predicted fragments for level C and D candidates |

The `final_results/` folder is made next to the output table.

## Batch processing

Process all MGF files in a folder, one result table per file:

```bash
mkdir -p results
for f in data/*.mgf; do
    name=$(basename "$f" .mgf)
    annomass headless --mgf "$f" --out "results/${name}.xlsx" --top-n 3
done
```

On a remote server, run long jobs with `nohup` so they continue after you log out:

```bash
nohup annomass headless --mgf big.mgf --out big_results.xlsx > run.log 2>&1 &
tail -f run.log
```
