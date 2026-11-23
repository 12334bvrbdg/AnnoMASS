# Model and data files

`download_data.sh` (called by `install.sh`) downloads every file below from the GitHub Release
of this repository and puts it in place. `ANNOMASS_HOME` defaults to `~/AnnoMASS`;
`<app>` is the AnnoMASS folder you downloaded. Checksums: `SHA256SUMS.txt` on the Release page.

## Files produced by AnnoMASS
| File(s) | Content | Put at |
|---|---|---|
| `references_index_{positive,negative}_spec2vec.bin`, `references_spectrums_{positive,negative}.pickle` | Experimental flavonoid spectral library and its search index | `<app>/data/` |
| `MassSpecGym_DreaMS_flavonoid_only.hdf5` | Flavonoid reference library with spectral embeddings | `<app>/data/reference_libraries/` |
| `spec2vec_spectraverse_{positive,negative}.model` + two `.npy` files each | Spectral-similarity models | `<app>/model/` |
| `FlavonoidStructureDB-dedup.csv` | Flavonoid structure library | `<app>/Stru/` |
| `flavonoid_candidates_final.csv` | Generated flavonoid structures | `$ANNOMASS_HOME/external/deepmet/final_results/` |
| `CFIL_extended.csv`, `TPIL_extended.xlsx` | Diagnostic fragment libraries | `$ANNOMASS_HOME/external/AnnoSM/Library/` |

## Pretrained weights of other projects (copies, unchanged)
| File | Original source | Put at |
|---|---|---|
| `ssl_model.ckpt`, `embedding_model.ckpt` | DreaMS, Zenodo record 10997887 | `$ANNOMASS_HOME/external/DreaMS/dreams/models/pretrained/` |
| `msg_all.zip` (folder `msg_all`: `gen/best.ckpt`, `inten_contr/best.ckpt`) | ICEBERG MassSpecGym-trained model, linked in the ms-pred README | unzip in `$ANNOMASS_HOME/external/ms-pred/models/` |
| ABCoRT `model_dict/best_model.pth` | part of the ABCoRT git repository (arrives with the installer) | `$ANNOMASS_HOME/external/ABCoRT/model_dict/` |
