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

## References

1. Bushuiev, R. et al. Self-supervised learning of molecular representations from millions of
   tandem mass spectra using DreaMS. *Nature Biotechnology* **44**, 630–640 (2026).
   <https://doi.org/10.1038/s41587-025-02663-3>
2. Bushuiev, R. et al. MassSpecGym: a benchmark for the discovery and identification of
   molecules. *Advances in Neural Information Processing Systems* **37**, 110010–110027 (2024).
   <https://doi.org/10.48550/arXiv.2410.23326>
3. Huber, F. et al. Spec2Vec: improved mass spectral similarity scoring through learning of
   structural relationships. *PLoS Computational Biology* **17**, e1008724 (2021).
   <https://doi.org/10.1371/journal.pcbi.1008724>
4. Huber, F. et al. matchms – processing and similarity evaluation of mass spectrometry data.
   *Journal of Open Source Software* **5**, 2411 (2020). <https://doi.org/10.21105/joss.02411>
5. Goldman, S., Li, J. & Coley, C. W. Generating molecular fragmentation graphs with
   autoregressive neural networks. *Analytical Chemistry* **96**, 3419–3428 (2024).
   <https://doi.org/10.1021/acs.analchem.3c04654>
6. Wang, R. et al. Neural spectral prediction for structure elucidation with tandem mass
   spectrometry. *bioRxiv* 2025.05.28.656653 (2025). <https://doi.org/10.1101/2025.05.28.656653>
7. Loeffler, H. H. et al. Reinvent 4: modern AI-driven generative molecule design.
   *Journal of Cheminformatics* **16**, 20 (2024). <https://doi.org/10.1186/s13321-024-00812-5>
8. Schymanski, E. L. et al. Identifying small molecules via high resolution mass spectrometry:
   communicating confidence. *Environmental Science & Technology* **48**, 2097–2098 (2014).
   <https://doi.org/10.1021/es5002105>

Confidence levels A–D follow the MSI levels of reference 8.

## Reference data

The spectral libraries of AnnoMASS were compiled from public data: MassSpecGym, SpectraVerse,
and the public reference spectra collected by DeepMASS2. We thank everyone who shares MS/MS
spectra in public libraries.
