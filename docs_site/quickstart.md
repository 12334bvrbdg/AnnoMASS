---
description: Annotate your first flavonoid MS/MS file with AnnoMASS in five minutes, using the GUI, the command line or the web version.
---

# Quick start

This page runs AnnoMASS on a small example file. It contains 5 MS/MS spectra of flavonoid
standards.

[:material-download: Download example_5_spectra.mgf](assets/example_5_spectra.mgf){ .md-button .md-button--primary }

=== "Desktop GUI"

    1. Start AnnoMASS: type `annomass` in a terminal, or use the **AnnoMASS** icon.
    2. Wait until the **Open** button becomes active. The models load in the background.
    3. Click **Open** and choose `example_5_spectra.mgf`. The 5 spectra appear in the list
       on the left.
    4. Click **Run**.
    5. When the run is finished, the **Results** tab shows the candidates. Click a row to see
       the spectrum and the structure.
    6. Click **Save Results (Excel)** to save the table.

    Details: [Desktop GUI](gui.md).

=== "Command line"

    ```bash
    annomass headless --mgf example_5_spectra.mgf --out example_results.xlsx --top-n 3
    ```

    Results:

    - `example_results.xlsx`: the result table
    - `final_results/figures/`: spectrum comparison plots and predicted fragments

    Details: [Command line](cli.md).

=== "Web"

    1. Go to <https://annomass.com>.
    2. Open **Annotate** and upload `example_5_spectra.mgf`.
    3. Start the run and download the result table when it is finished.

    Details: [Web version](web.md).

## What you should see

All 5 spectra are annotated at level <span class="level level-a">A</span>
(match to an experimental reference spectrum). The top candidates should be:

| Precursor *m/z* | Top candidate | Formula | Level | Score |
|---|---|---|:---:|---|
| 255.0653 | Daidzein | C15H10O4 | A | 0.939 |
| 271.0602 | Genistein | C15H10O5 | A | 0.933 |
| 287.0548 | Luteolin | C15H10O6 | A | 0.836 |
| 303.0495 | Quercetin | C15H10O7 | A | 0.944 |
| 319.0447 | Myricetin | C15H10O8 | A | 0.933 |

Small differences in the third decimal of the score are normal between machines.

For each spectrum the table lists the top candidates (3 by default), with:

- **Annotation basis**: the level (A–D) and the kind of evidence
- **Annotation score**: the score of that level (compare scores only within the same level)
- **Candidate rank**: 1 = best candidate for this spectrum

See [Input and output](input-output.md) for every column.
