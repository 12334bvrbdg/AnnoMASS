---
description: Step-by-step tutorial for the AnnoMASS desktop GUI. Load spectra, choose settings, run the annotation, inspect results and export tables and figures.
---

# Desktop GUI

The GUI is the easiest way to look at results in detail: spectra, structures, predicted
fragments and the evidence behind each candidate.

!!! info "Screenshots"
    Screenshots will be added in a later version of this page.

## Start the GUI

=== "Linux"

    ```bash
    annomass
    ```

    or click **AnnoMASS** in the application menu.

=== "Windows (WSL2)"

    Double-click the **AnnoMASS** icon on the desktop.

A splash window appears first. Models and libraries load in the background, which takes
about 10–60 seconds. The **Open** and **Run** buttons become active when loading is finished.

## Main window

| Area | What it does |
|---|---|
| Toolbar | **Open** loads spectrum files. **Run** starts the annotation. **Settings** opens the parameter page. |
| Spectrum list (left) | All loaded spectra. Click one to preview it. |
| **Results** tab | Result table, spectrum viewer, structure viewer, predicted fragments and embedding map. |
| **Settings** tab | All run parameters. |
| **Manual / Agent** switch (top) | **Manual** is the normal analysis view. **Agent** opens an optional chat assistant (see [below](#agent-mode-optional)). |

## Step 1: Load spectra

1. Click **Open**.
2. Choose one or more files (`.mgf`, `.msp`). Hold ++ctrl++ to choose several files.
3. The spectra appear in the list on the left. Click a spectrum to see it in the viewer.

See [Input and output](input-output.md#input-file) for the file requirements.

## Step 2: Check the settings (optional)

The default settings work for most data. Click **Settings** to change them.
**Restore defaults** resets all values. **← Back to Results** returns to the results.

### Run modules

Four checkboxes, **A**, **B**, **C** and **D**, select the levels to run (all are on by default).
A level that is unticked is not computed. Examples:

| Ticked | Use |
|---|---|
| A B C D | Normal analysis (default) |
| A | Only fast library matching |
| B C | Structure-library inference only, without library matching |
| A B C | Skip generated structures (faster) |

### Library match

| Setting | Meaning | Default |
|---|---|---|
| Min similarity | Lowest library similarity for a hit to be kept (0–1) | 0.70 |
| CHO only | Keep only compounds made of C, H and O | On |

### Fragment simulation

These settings describe your instrument. They are used for predicted spectra (levels C and D).

| Setting | Meaning | Default |
|---|---|---|
| Collision energy (NCE) | Collision energy of your MS/MS method | 30 |
| Instrument | Orbitrap, QTOF, IT-FT, or Unknown | Unknown |

### Level D thresholds

Two thresholds for generated structures: one score must reach the primary threshold (0.5)
and the other the secondary threshold (0.4). See [Algorithm](algorithm.md#level-d).

### ICE peak cleaning

How the spectra are cleaned before they are compared with predicted spectra.

| Setting | Meaning | Default |
|---|---|---|
| Min relative intensity (%) | Peaks below this % of the base peak are removed | 1 |
| Strongest peaks | Keep at most this many peaks (0 = keep all) | 80 |

### Optional refinements

| Setting | Meaning | Default |
|---|---|---|
| Substituent-position check | Reorders close candidates with flavonoid diagnostic fragments | On |
| Predict retention time | Adds a predicted retention time to each candidate (for information only, not used for ranking) | Off |
| Formula pre-filter | Removes candidates whose formula does not fit the spectrum | Off |

## Step 3: Run

Click **Run**. The progress bar and the status line show the current step.

As a guide, on a 7-core CPU server without GPU, 10 spectra took about 1 minute with levels
A–C, and level D added about 10 minutes. A GPU makes levels C and D much faster.

Spectra annotated at level A finish fastest. Levels C and D need predicted spectra, which
take most of the time.

## Step 4: Look at the results

The **Results** tab opens when the run is finished.

### Result table

One row per candidate. The most important columns:

- **Level** and **MSI level**: how the candidate was found (A–D) and its confidence level
- **Level score**: score of that level; compare scores only within the same level
- **Compound name**, **Formula**, **SMILES**

Rows are coloured by level.

**Top-K per spectrum** sets how many candidates are shown per spectrum (default 3). Change the
number and click **Apply Filter**.

**Show D (CLM)** shows or hides the generated structures (level D). They are extra
suggestions and are kept apart from the A–C results.

### Viewers

Click a row in the table:

- **Spectrum viewer**: your spectrum (top) against the reference or predicted spectrum (bottom).
  Matched peaks are highlighted.
- **Structure**: the structure of the candidate.
- **Predicted fragment peaks**: for levels C and D, a table of predicted fragments. Click a
  row, or a blue peak in the spectrum, to see the fragment structure. **Full structure**
  returns to the whole molecule. **Save fragment** saves the fragment image.
- **Embedding map**: a 2D map of your spectrum among the library spectra. Close points have
  similar spectra.

## Step 5: Save

| Button | What it saves |
|---|---|
| **Save Results (Excel)** | The result table as `.xlsx` (see [output columns](input-output.md#result-table)) |
| **Save Figures** | Spectrum comparison plots and fragment figures for the candidates |
| **Save Project** | Everything (spectra, settings, results) as an `.annomass` file |
| **Load Project** | Opens a saved `.annomass` file, without running again |
| **Clear** | Clears the results |
| **Unload Spectra** | Removes the loaded spectra |

!!! tip
    Save a project after long runs. You can reopen it later and export again without
    repeating the calculation.

## Agent mode (optional)

Switch from **Manual** to **Agent** at the top of the window to open a chat assistant. It can
read your results and answer questions, for example "Which spectra are annotated at level A?"
It needs a large-language-model API key.

1. Switch to **Agent** and click **Settings…** in the agent panel.
2. Enter the API address, model name and key. Click **Test Connection**.
3. Ask your question and click **Send**.

The agent is not needed for annotation. Nothing is sent to an outside service unless you use
the agent.
