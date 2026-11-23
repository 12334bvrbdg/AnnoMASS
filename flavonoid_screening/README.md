# Flavonoid screening

`screen_flavonoids.py` picks the flavonoids out of any structure table. It uses the rules that
built the AnnoMASS flavonoid structure library (`FlavonoidStructureDB-dedup.csv`):

1. only C, H and O (the largest fragment is used, so counter-ions are ignored);
2. at least 15 carbon atoms (C6-C3-C6 skeleton);
3. a flavonoid core: one of the core SMARTS in the script, or a flavonoid subclass recognized
   by `core/flavclass_filter.py` (flavone, flavonol, flavanone, flavanonol, isoflavone,
   isoflavanone, isoflavan, chalcone, dihydrochalcone, aurone, anthocyanidin, flavan-3-ol,
   flavan, pterocarpan);
4. duplicates removed by InChIKey.

The output has the same columns as the AnnoMASS structure library
(`Title, Formula, SMILES, InChIkey, Exact mass, Database IDs`), so it can replace or extend
`Stru/FlavonoidStructureDB-dedup.csv`.

## Requirements

Python 3.9+ with RDKit and pandas. The AnnoMASS `main` environment already has them:

```bash
~/AnnoMASS/envs/main/bin/python flavonoid_screening/screen_flavonoids.py --help
```

Without AnnoMASS: `pip install rdkit pandas`, then run the script from the repository folder
(it imports `core/flavclass_filter.py`).

## Structure databases you can screen

| Database | Download | Size |
|---|---|---|
| DeepMASS2 structure database (the source of the AnnoMASS library) | [`DeepMassStructureDB-v1.1.csv`](https://github.com/hcji/DeepMASS2_GUI/releases/download/v0.99.1/DeepMassStructureDB-v1.1.csv), DeepMASS2 release v0.99.1 | 116 MB, 559,589 structures |
| PubChem, all compounds | [`CID-SMILES.gz`](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) | about 1.5 GB, over 100 million structures |
| COCONUT natural products | [COCONUT download page](https://coconut.naturalproducts.net/download) (CSV) | about 700,000 structures |

## Examples

DeepMASS2 structure database (about 30 s with 6 processes):

```bash
curl -LO https://github.com/hcji/DeepMASS2_GUI/releases/download/v0.99.1/DeepMassStructureDB-v1.1.csv
python flavonoid_screening/screen_flavonoids.py \
    --input DeepMassStructureDB-v1.1.csv --out flavonoids_deepmass.csv --workers 6
# 559,589 structures read, 22,993 unique flavonoids written to flavonoids_deepmass.csv
```

PubChem (no header; column 0 = CID, column 1 = SMILES):

```bash
curl -LO https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz
python flavonoid_screening/screen_flavonoids.py \
    --input CID-SMILES.gz --no-header --id-col 0 --smiles-col 1 \
    --out flavonoids_pubchem.csv --workers 16
```

Any other table: the SMILES column is found by name (a column whose name contains "smiles");
set it with `--smiles-col` if needed. CSV, TSV and `.gz` files are accepted.

## Use the result in AnnoMASS

To annotate against your own flavonoid library, replace the structure library (keep a copy of
the original):

```bash
cp Stru/FlavonoidStructureDB-dedup.csv Stru/FlavonoidStructureDB-dedup.original.csv
cp flavonoids_deepmass.csv Stru/FlavonoidStructureDB-dedup.csv
```

## Relation to the released library

The released `FlavonoidStructureDB-dedup.csv` (18,254 structures) was screened from the
DeepMASS2 structure database and a second structure collection, and a small number of
reference compounds were added by hand. Running the script on the released library keeps
17,959 of its 18,254 rows (98.4%).
