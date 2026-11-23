# AnnoMASS installer

Linux:    `bash install.sh`  -> then type `annomass`, or use the application-menu entry.
Windows:  double-click `Install_AnnoMASS.bat` (needs WSL2) -> desktop icon "AnnoMASS".
Command line: `annomass headless --mgf in.mgf --out out.xlsx`

`install.sh` builds the environments, fetches the third-party code at fixed commits and
downloads the model and data files (`download_data.sh`, about 3.7 GB). It can be run again
at any time; finished parts are skipped.

## Layout created under `~/AnnoMASS` (override with `ANNOMASS_HOME`)
```
AnnoMASS/
  conda/                      Miniforge
  envs/main                   main program and GUI                     (Python 3.10)
  envs/spectrum-retrieval     spectral-library search                  (Python 3.11)
  envs/spectrum-prediction    fragment/spectrum prediction             (Python 3.11)
  external/DreaMS ms-pred ABCoRT AnnoSM deepmet
```

## Why three environments
DreaMS pins `numpy==1.25.0` (and numba 0.58); ICEBERG needs `numpy>=2`.

## Pinned commits
The installer checks out the exact commit of each third-party project used for the paper
(e.g. DreaMS `e37a47d`) and applies the small patches in `patches/`, so every user runs the
same code.

## Model and data files
See `WEIGHTS.md`. Options: `ANNOMASS_SKIP_DOWNLOAD=1 bash install.sh` skips the download;
`bash download_data.sh` downloads (or resumes) it later.
