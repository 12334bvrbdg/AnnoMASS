---
description: Open-source tools, models and data that AnnoMASS builds on, and how to credit them.
---

# Acknowledgements

AnnoMASS is built on open-source tools, models and data from other groups. We thank their
authors. If you publish results obtained with AnnoMASS, please also cite the original papers
of the components below (listed on each project page).

## Components used in the pipeline

| Component | Role in AnnoMASS | Project |
|---|---|---|
| **DreaMS** | Spectral embedding for library matching (level A) | [pluskal-lab/DreaMS](https://github.com/pluskal-lab/DreaMS) |
| **MassSpecGym** | Reference spectra for the level A library | [pluskal-lab/MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) |
| **DeepMASS2** | Structure-library inference and reference spectra (levels B and C) | [hcji/DeepMASS2_GUI](https://github.com/hcji/DeepMASS2_GUI) |
| **Spec2Vec** | Learned spectral similarity used in structure inference | [iomega/spec2vec](https://github.com/iomega/spec2vec) |
| **matchms** | Spectrum processing and modified cosine | [matchms/matchms](https://github.com/matchms/matchms) |
| **ICEBERG** (ms-pred) | Fragmentation and spectrum prediction (levels C and D) | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) |
| **DeepMet / CLM** | Chemical language model framework for generated structures (level D) | [skinniderlab/CLM](https://github.com/skinniderlab/CLM) |
| **REINVENT4** | Pre-trained prior of the chemical language model | [MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4) |
| **AnnoSM** | Diagnostic fragment library for the substituent-position check | [YangHuaLab/AnnoSM](https://github.com/YangHuaLab/AnnoSM) |
| **ABCoRT** | Retention-time prediction (optional) | [RiverCCC/ABCoRT](https://github.com/RiverCCC/ABCoRT) |
| **FIDDLE** | Formula pre-filter (optional) | see its original publication |
| **RDKit** | Chemistry toolkit (structures, InChIKeys, substructure search) | [rdkit/rdkit](https://github.com/rdkit/rdkit) |

AnnoMASS installs each component at a fixed version. Small local changes needed for the
integration are kept as patch files in `release_files/installer/patches/`. Each component keeps
its own license; see the license file in its folder under `~/AnnoMASS/external/`.

## Confidence levels

The confidence levels follow:

> Schymanski, E. L. et al. Identifying small molecules via high resolution mass spectrometry:
> communicating confidence. *Environmental Science & Technology* **48**, 2097–2098 (2014).

## Reference data

The spectral libraries of AnnoMASS were compiled from public data: MassSpecGym, SpectraVerse,
and the public reference spectra collected by DeepMASS2. We thank everyone who shares MS/MS
spectra in public libraries.
