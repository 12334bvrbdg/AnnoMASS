---
description: Install AnnoMASS locally on Linux or on Windows (WSL2). System requirements, installer, model and data files, and how to check the installation.
---

# Installation

AnnoMASS runs on **Linux** (Ubuntu 22.04 or later recommended) and on **Windows 10/11 through
WSL2**. macOS is not supported.

!!! tip "No time to install?"
    The [web version](web.md) runs the same pipeline in your browser for small batches.

## System requirements

| Item | Minimum | Recommended |
|---|---|---|
| Operating system | Linux x86-64, or Windows 10/11 with WSL2 | Ubuntu 22.04 / 24.04 |
| CPU | 4 cores | 8 cores or more |
| Memory (RAM) | 16 GB | 32 GB |
| Disk space | 30 GB free | 50 GB free (SSD) |
| GPU | Not required | NVIDIA GPU with a recent driver |
| Other | `git`, `curl`, internet connection during installation | |

!!! note "GPU"
    An NVIDIA GPU is recommended. Without a GPU, AnnoMASS runs on the CPU, and the
    fragment-prediction steps (levels C and D) are slower.

## Overview

Installation has three steps:

1. Get the AnnoMASS code.
2. Run the installer. It builds the Python environments and downloads the third-party code.
3. The installer downloads the model and data files.

The installer creates everything under `~/AnnoMASS` (you can change this with
`ANNOMASS_HOME`). It does not change your system Python or other conda environments.

```text
~/AnnoMASS/
├── conda/                     Miniforge (private copy)
├── envs/
│   ├── main/                  main program and GUI       (Python 3.10)
│   ├── spectrum-retrieval/    spectral-library search    (Python 3.11)
│   └── spectrum-prediction/   fragment prediction        (Python 3.11)
└── external/                  third-party code at fixed versions
```

Three environments are needed because two of the components need different, incompatible
versions of `numpy`.

## Step 1: Get the code

```bash
git clone https://github.com/12334bvrbdg/AnnoMASS.git
cd AnnoMASS
```

You can also download the ZIP file from the GitHub page and unpack it.

!!! warning "Use a path without spaces or non-ASCII characters"
    Some scientific tools fail on paths such as `~/我的文件/AnnoMASS` or `~/My Files/AnnoMASS`.

## Step 2: Run the installer

=== "Linux"

    ```bash
    bash release_files/installer/install.sh
    ```

    To install somewhere else, for example on a data disk:

    ```bash
    ANNOMASS_HOME=/data/AnnoMASS bash release_files/installer/install.sh
    ```

=== "Windows (WSL2)"

    1. Install WSL2 with Ubuntu. Open **PowerShell as administrator** and run:

        ```powershell
        wsl --install -d Ubuntu
        ```

        Restart Windows when asked, then open **Ubuntu** once to create a user name and password.

    2. If you have an NVIDIA GPU, install the normal **Windows** NVIDIA driver.
       Do not install a separate Linux driver inside WSL.

    3. Put the AnnoMASS folder inside the Linux file system (for example `~/AnnoMASS-app`),
       not under `C:\`. Files under `/mnt/c/` are much slower.

    4. Double-click `release_files\installer\Install_AnnoMASS.bat`, or run inside Ubuntu:

        ```bash
        bash release_files/installer/install.sh
        ```

    The installer adds an **AnnoMASS** icon to your Windows desktop.

The first run takes 20–60 minutes, depending on your network. It is safe to run the installer
again: finished parts are skipped.

At the end, the installer downloads the model and data files (step 3) and checks that
all of them are in place.

## Step 3: Model and data files

The installer downloads the model weights, spectral libraries and structure libraries
(about 3.7 GB) from the [GitHub Release](https://github.com/12334bvrbdg/AnnoMASS/releases)
and puts each file in place. If the download stops, run it again; finished files are skipped:

```bash
bash release_files/installer/download_data.sh
```

When everything is in place, the installer prints `All model/data files present.`

??? note "Manual download"
    You can also download the files from the Release page yourself and copy them here.
    `<app>` is the AnnoMASS folder from step 1.

    | File | Put it in |
    |---|---|
    | `references_index_positive_spec2vec.bin`, `references_index_negative_spec2vec.bin`, `references_spectrums_positive.pickle`, `references_spectrums_negative.pickle` | `<app>/data/` |
    | `MassSpecGym_DreaMS_flavonoid_only.hdf5` | `<app>/data/reference_libraries/` |
    | `spec2vec_spectraverse_positive.model`, `spec2vec_spectraverse_negative.model` and their `.npy` files | `<app>/model/` |
    | `FlavonoidStructureDB-dedup.csv` | `<app>/Stru/` |
    | `flavonoid_candidates_final.csv` | `~/AnnoMASS/external/deepmet/final_results/` |
    | `CFIL_extended.csv`, `TPIL_extended.xlsx` | `~/AnnoMASS/external/AnnoSM/Library/` |
    | `ssl_model.ckpt`, `embedding_model.ckpt` | `~/AnnoMASS/external/DreaMS/dreams/models/pretrained/` |
    | `msg_all.zip` (unzip it here) | `~/AnnoMASS/external/ms-pred/models/` |

    Check the downloads with `sha256sum -c SHA256SUMS.txt`.

The DreaMS and ICEBERG weights are copies of the files published by their authors
(DreaMS: Zenodo [10997887](https://zenodo.org/records/10997887); ICEBERG: the MassSpecGym-trained
`msg_all` model linked in the [ms-pred README](https://github.com/coleygroup/ms-pred)).
Please cite the original papers when you use them.

## Check the installation

Run the 5-spectrum example from the command line:

```bash
annomass headless --mgf examples/example_5_spectra.mgf --out test_result.xlsx
```

The run takes a few minutes. It should end with a line like
`[headless] saved N rows to test_result.xlsx`.
Then start the GUI:

```bash
annomass
```

If `annomass` is not found, open a new terminal, or add `~/.local/bin` to your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

## Update

```bash
cd AnnoMASS
git pull
bash release_files/installer/install.sh
```

## Uninstall

Delete the install folder and the two launcher files:

```bash
rm -rf ~/AnnoMASS
rm -f ~/.local/bin/annomass ~/.local/share/applications/annomass.desktop
```

Then delete the AnnoMASS code folder from step 1.

## Third-party components

AnnoMASS builds on open-source tools from other groups. The installer fetches each one at a
fixed version, so all users run the same code. Please also cite these tools when you publish
results. The full list is on the [Acknowledgements](acknowledgements.md) page.
