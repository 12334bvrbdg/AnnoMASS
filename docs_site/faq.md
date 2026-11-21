---
description: Frequently asked questions and troubleshooting for AnnoMASS installation, GUI, command line and results.
---

# FAQ

## Results

??? question "A spectrum has no row in the result table. Why?"
    Possible reasons:

    - It is a negative-mode spectrum. AnnoMASS uses positive mode only.
    - It has fewer than 3 fragment peaks (see [Input](input-output.md#requirements)).
    - No candidate passed any level. This often means the compound is not a flavonoid.

??? question "Can I compare annotation scores between spectra?"
    Only within the same level. An A score of 0.75 and a C score of 0.75 mean different things.
    Use the level first, then the score.

??? question "The top candidate is an isomer of the true compound."
    Some isomers give almost the same MS/MS spectrum. Check the top 3 candidates and use
    retention time or standards. Level C results in particular should be read as
    "one of these isomers".

??? question "Should I report level D structures?"
    Report them as MSI Level 4 structure hypotheses, not as identified compounds.

??? question "Why are the results slightly different from the web version?"
    Small differences in the third decimal can come from different hardware (CPU or GPU).
    The ranking should be the same. The web version runs levels A–C by default; level D is
    optional there.

## Installation

??? question "`annomass: command not found`"
    Open a new terminal, or add `~/.local/bin` to your `PATH`:

    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
    ```

??? question "The installer says files are missing."
    The model and data files must be downloaded by hand. See
    [Installation, step 3](installation.md#step-3-model-and-data-files). Run the installer
    again after copying; it lists what is still missing.

??? question "The GUI does not open in WSL2."
    Update WSL (`wsl --update` in PowerShell) so that Linux windows can be shown, then restart
    WSL (`wsl --shutdown`). The command line works without a window.

??? question "Out of memory"
    Close other programs, run fewer spectra at a time, or skip level D (`--modules ABC`).

## Other

??? question "Can I use AnnoMASS for other compound classes?"
    Not with the current libraries and models. They are made for flavonoids.

??? question "Where do I report a bug?"
    Open an issue on the GitHub repository. Please attach the log and, if possible, a small
    MGF file that shows the problem.
