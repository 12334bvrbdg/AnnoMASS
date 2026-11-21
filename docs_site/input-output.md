---
description: Input file requirements (positive-mode MGF) and the meaning of every column in the AnnoMASS result table, including annotation levels A–D and MSI levels.
---

# Input and output

## Input file

AnnoMASS reads MS/MS spectra in **MGF** format. The GUI also reads **MSP**.

### Requirements

| Item | Requirement |
|---|---|
| Ion mode | **Positive** (`[M+H]+`). Negative-mode spectra are skipped. |
| Precursor | `PEPMASS` must be given. Use 4 or more decimals if you can. |
| Fragment peaks | At least 3 fragment peaks (≥1% of the base peak, more than 2 Da from the precursor). Spectra with fewer peaks are skipped. |
| Spectrum type | One MS/MS spectrum per `BEGIN IONS ... END IONS` block. Centroid data are best. |
| Compound class | Flavonoids. Other compound classes will get no result or wrong results. |

### Example

```text
BEGIN IONS
TITLE=Sample1_ID10_RT16.29
PEPMASS=255.06529
ION=[M+H]+
CHARGE=1+
RTINMINUTES=16.287
55.02415	26
91.05678	138
137.02411	552
...
END IONS
```

- `TITLE` is copied into the result table (column **Spectrum IDs**). Use a unique title for
  each spectrum, for example sample name + feature ID + retention time.
- `ION` (or `ADDUCT`) is optional. If it is missing, `[M+H]+` is assumed.
- `RTINMINUTES` is optional and is not used for scoring.

### Exporting MGF from other software

Most feature-detection software can export MS/MS spectra as MGF, for example MS-DIAL
(*Export → Spectra → MGF*), MZmine (*Export → MGF*) or vendor software. Export the
**positive-mode** MS/MS spectra of your features, one spectrum per feature.

!!! tip "Precursor precision"
    If `PEPMASS` has only 2–3 decimals, AnnoMASS widens the mass tolerance a little to
    match. Still, more decimals give fewer wrong candidates.

## Result table

The result table (`.xlsx` or `.csv`) has one row per candidate. Candidates of the same spectrum
are sorted by rank.

| Column | Meaning |
|---|---|
| **Compound name** | Name of the candidate. When no common name is known (for example for generated structures), the InChIKey or SMILES is shown. |
| **Formula** | Molecular formula |
| **SMILES** | Structure |
| **InChIKey** | Structure identifier, computed from the SMILES |
| **Annotation basis** | Level and kind of evidence (see below) |
| **Candidate rank** | 1 = best candidate of this spectrum |
| **Annotation score** | Score of the level (0–1). Compare scores only within the same level. |
| **Precursor m/z** | Precursor *m/z* of the spectrum |
| **Spectrum IDs** | `TITLE` of the spectrum (or spectra) that gave this candidate |

### Annotation basis

| Value | Meaning | MSI level |
|---|---|:---:|
| `A | Experimental spectral-library matching` | Your spectrum matches an experimental reference spectrum of this compound. | 2 |
| `B | Structure-library inference + experimental spectral evidence` | The structure is inferred from similar library spectra, and an experimental spectrum of the candidate confirms it. | 2 |
| `C | Structure-library inference + predicted fragmentation evidence` | The structure is inferred from similar library spectra and supported by a predicted spectrum. | 3 |
| `D | Generated-structure inference + predicted fragmentation evidence` | A generated structure that is not in the structure database, supported only by a predicted spectrum. | 4 |

### How to read the levels

- **A and B (MSI Level 2)**: probable structure. Still check it with a standard if the
  compound is important for your study.
- **C (MSI Level 3)**: the compound class and the substitution pattern are usually right, but
  the exact isomer is uncertain. Look at the top 3 candidates, not only rank 1.
- **D (MSI Level 4)**: a structure hypothesis only. Generated structures have no
  stereochemistry. "Not in the database" does not mean "a new compound".
- **No row for a spectrum**: no candidate passed any level. AnnoMASS gives no answer rather
  than a weak one.

!!! warning "Isomers"
    Some isomers give almost the same MS/MS spectrum (for example a flavanone and its
    open-chain chalcone isomer). Use retention time and standards to tell them apart.

### Each spectrum once, or merged?

By default, if several spectra give the same compound (for example the same compound in
several samples), they are merged into one row, and **Spectrum IDs** lists all of them.
To report each spectrum separately, use `--per-spectrum-dedup` in the
[command line](cli.md).

## Figures

The GUI (**Save Figures**) and the command line save:

| Folder | Content |
|---|---|
| `figures/spectra_comparison/` | Mirror plots: your spectrum (top) and the reference or predicted spectrum (bottom) |
| `figures/iceberg_fragments/` | Predicted fragment peaks and fragment structures (levels C and D) |
