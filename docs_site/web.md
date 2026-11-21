---
description: Use AnnoMASS in the browser at https://annomass.com. Upload an MGF file, run the annotation, search the reference libraries and predict spectra, with no installation.
---

# Web version

**Address: <https://annomass.com>**

The web version runs the AnnoMASS pipeline on our server. Nothing needs to be installed.
The page is available in English and Chinese (switch at the top).

| Good for | Use the local version instead when |
|---|---|
| A first try | You have more than 100 spectra |
| Small batches | Your data are confidential |
| Searching the reference libraries | You need the spectrum and fragment viewers of the GUI |

## Limits

- Up to **100 spectra** and **20 MB** per file
- Up to **3 analyses** run at the same time. Others wait in line.
- The server runs on CPU only, so runs take longer than on a local computer with a GPU.

## Annotate

1. Open the **Annotate** tab.
2. Upload an MGF file, or click **Load example** to use the 5 flavonoid standards.
3. Choose a **Method**:

    | Method | Levels |
    |---|---|
    | Full pipeline (default) | A → B → C |
    | Embedding match only | A |
    | Similarity inference only | B → C |

4. Optional: tick **Include level D** to add generated structures. This is much slower.
5. Click **Run annotation**. The progress is shown below the button. **Stop** cancels the run.
6. When the run is finished:
    - The table shows the top 3 candidates per spectrum.
    - The gallery shows the candidate structures.
    - **Download full result table (xlsx)** saves the full table
      (see [output columns](input-output.md#result-table)).

## Library search

Search the reference data without uploading a spectrum.

1. Open **Library search**.
2. Type a compound name, InChIKey, formula, SMILES or precursor *m/z*. **Search by: Auto detect**
   recognises the type for you.
3. Choose the libraries:
    - **Embedding library**: experimental spectra used at level A
    - **Reference library**: experimental spectra used at level B
    - **Generated compounds**: the generated flavonoid structures used at level D
4. Click a row to see the structure and the spectrum.
5. Download the hits as a table (CSV) or as spectra (MGF).

### Predict a spectrum

Below the search results you can predict the MS/MS spectrum of the selected compound.
Choose the **Adduct** and **Collision energy (NCE)**, then click
**Predict spectrum of the selected compound**. The predicted spectrum can be downloaded as MGF.

## Agent chat

The **Agent chat** tab is an optional assistant. It works on the file uploaded in
**Annotate**. For example, ask: "Run the annotation and summarise the top candidates."

You need your own large-language-model API key. Choose the provider, enter the key and choose
a model. The key stays in your browser session and is not stored on the server.

## Privacy

Uploaded files and results are saved in temporary folders on the server and are not shared
with other users. Do not upload confidential data. For confidential data, use the
[local installation](installation.md).
