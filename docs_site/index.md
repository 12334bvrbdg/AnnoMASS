---
description: AnnoMASS annotates flavonoid MS/MS spectra in confidence tiers (A–D, MSI Level 2–4). Documentation for local installation, the desktop GUI, the command line and the web version.
---

# AnnoMASS

**AnnoMASS** (Annotation of Mass Spectrometry Spectra) annotates **flavonoid** structures
from positive-mode **LC-MS/MS** spectra.

Each spectrum goes through a series of evidence levels. The first level is the most reliable.
A spectrum stops at the **first level that passes**, and every result gets a clear confidence
label (A–D) mapped to the Metabolomics Standards Initiative (MSI) levels.

| Level | Evidence | MSI level |
|:---:|---|:---:|
| <span class="level level-a">A</span> | Match to an experimental reference spectrum (spectral-library matching) | Level 2 |
| <span class="level level-b">B</span> | Structure-library inference, supported by an experimental reference spectrum | Level 2 |
| <span class="level level-c">C</span> | Structure-library inference, supported by a predicted spectrum | Level 3 |
| <span class="level level-d">D</span> | Generated structure, supported by a predicted spectrum | Level 4 |

See [Algorithm](algorithm.md) for how each level works.

## Three ways to use AnnoMASS

<div class="grid cards" markdown>

-   :material-web: **Web version**

    ---

    No installation needed. Upload an MGF file in the browser. Good for a first try and
    for small batches.

    [:octicons-arrow-right-24: https://annomass.com](web.md)

-   :material-monitor: **Desktop GUI**

    ---

    Interactive analysis on your own computer: spectrum viewer, structures, predicted
    fragments, and Excel export.

    [:octicons-arrow-right-24: GUI tutorial](gui.md)

-   :material-console: **Command line (CLI)**

    ---

    Batch processing and scripting, on a workstation or a server.

    [:octicons-arrow-right-24: CLI tutorial](cli.md)

</div>

## Where to start

1. **Just want to try it?** Open the [web version](web.md) and upload the
   [5-spectrum example file](assets/example_5_spectra.mgf).
2. **Want to run it on your own data?** Follow [Installation](installation.md), then
   [Quick start](quickstart.md).
3. **Need to read the results?** See [Input and output](input-output.md).

## Scope

- Compound class: **flavonoids** (the libraries and models are made for flavonoids).
- Ion mode: **positive** only (`[M+H]+`). Negative-mode spectra are skipped.
- Input: MS/MS spectra in **MGF** format (MSP is also read by the GUI).
- AnnoMASS never assigns MSI Level 1. Level 1 needs a match to an authentic standard
  (retention time and MS/MS) measured in your own lab.

## Citation

The AnnoMASS paper is in preparation. Citation details will be added here.
