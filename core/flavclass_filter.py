# -*- coding: utf-8 -*-
"""
Flavonoid scaffold recognition module

Key fix (original version Bug):
  The original version used SMARTS such as 'O=c1oc2ccccc2cc1', writing C4(keto carbonyl)
  adjacent to O1(ring oxygen), but in the chromone ring C4 is not directly bonded to O1,
  causing all natural flavonoids to fail matching.
  Correct traversal order: C4(=O) - C3 - C2 - O1 - C8a - C4a - C4 (ring closure).

Subclasses covered:
  Flavone, Flavonol, Flavanone, Flavanonol, Isoflavone, Isoflavanone, Isoflavan,
  Chalcone, Dihydrochalcone, Aurone, Anthocyanidin, Flavan-3-ol (Catechin), Flavan,
  Pterocarpan and IsoflavanChroman.

ChromoneCore and Xanthone are kept as auxiliary "flavonoid-like" labels for
diagnostics, but they are not sufficient by themselves for is_flavonoid().
"""
from rdkit import Chem

CORE_FLAVONOID_CLASSES = {
    "Flavone", "Flavonol", "Flavanone", "Flavanonol", "Dihydroflavonol",
    "Isoflavone", "Isoflavanone", "Isoflavan", "Flavan", "Flavan3ol",
    "Chalcone", "Dihydrochalcone", "Aurone", "Anthocyanidin", "Pterocarpan",
    "IsoflavanChroman",
}
AUXILIARY_FLAVONOID_LIKE_CLASSES = {"ChromoneCore", "Xanthone"}

SMARTS = {
    # ── Flavone series ────────────────────────────────────────
    # Flavone: 2-phenylchromone, C4=O, C2=C3, Ph at C2
    "Flavone":         "O=c1cc(-c2ccccc2)oc2ccccc12",
    # Flavonol: Flavone + C3-OH
    "Flavonol":        "O=c1c(O)c(-c2ccccc2)oc2ccccc12",

    # ── Flavanone series ──────────────────────────────────────
    # Flavanone: C2-C3 saturated, Ph at C2
    "Flavanone":       [
        "O=C1CC(-c2ccccc2)Oc2ccccc21",
        "O=C1Cc2ccccc2OC1-c1ccccc1",
    ],
    # Flavanonol: Flavanone + C3-OH
    "Flavanonol":      [
        "O=C1C(O)C(-c2ccccc2)Oc2ccccc21",
        "OC1C(=O)c2ccccc2OC1-c1ccccc1",
    ],

    # ── Isoflavone series ─────────────────────────────────────
    # Isoflavone: 3-phenylchromone, Ph at C3, C2=C3
    "Isoflavone":      [
        "O=c1c(-c2ccccc2)coc2ccccc12",
        "O=c1oc2ccccc2c(-c2ccccc2)c1",
    ],
    # Isoflavanone: 3-phenylflavanone, Ph at C3, C2-C3 saturated
    "Isoflavanone":    "O=C1C(-c2ccccc2)COc2ccccc21",
    # Isoflavan: no carbonyl, Ph at C3
    "Isoflavan":       "C1(-c2ccccc2)c2ccccc2OCC1",

    # ── Other flavonoids ─────────────────────────────────────
    # Chalcone: open-chain C=C-C=O
    "Chalcone":        "O=C(/C=C/c1ccccc1)c1ccccc1",
    # Dihydrochalcone
    "Dihydrochalcone": "O=C(CCc1ccccc1)c1ccccc1",
    # Aurone: benzofuranone
    "Aurone":          "O=C1OC(=Cc2ccccc2)c2ccccc21",
    # Anthocyanidin: flavylium cation
    "Anthocyanidin":   "[OH+]=c1cc(-c2ccccc2)oc2ccccc12",
    # Flavan-3-ol / Catechin
    "Flavan3ol":       "OC1Cc2ccccc2OC1c1ccccc1",
    # Flavan: 2-phenyl-2,3-dihydrobenzopyran, no C4=O
    "Flavan":          "C1(-c2ccccc2)COc2ccccc21",
    # Xanthone
    "Xanthone":        "O=C1c2ccccc2Oc2ccccc21",
    # Generic ChromoneCore: catch-all, avoids missed matches
    "ChromoneCore":    "O=c1ccoc2ccccc12",
    # 3-phenylchroman / neoisoflavan (IsoflavanChroman): three six-membered rings,
    # distinct from isoflavans with 5-membered rings
    # Covers Isomucronulatol and other wing-bean-type compounds
    "IsoflavanChroman": "[CH]1([CH2]Oc2ccccc2[CH2]1)c1ccccc1",
    # Pterocarpan: 4-ring scaffold with 5-membered oxygen-containing ring
    # (benzofuran) and 6-membered oxygen-containing ring (chroman)
    # Covers Methylnissolin, Medicarpin, etc.
    "Pterocarpan": "C1COc2ccccc2C1Oc1ccccc1",
}

PATTS = {}
for k, values in SMARTS.items():
    if isinstance(values, str):
        values = [values]
    plist = []
    for v in values:
        p = Chem.MolFromSmarts(v)
        if p is not None:
            plist.append(p)
    if plist:
        PATTS[k] = plist

_cache = {}


def classify(smiles: str) -> set:
    """Return the set of flavonoid subclass names matched by this SMILES; empty set means non-flavonoid."""
    if not smiles:
        return set()
    smi = str(smiles).strip()
    if smi in _cache:
        return _cache[smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _cache[smi] = set()
        return set()
    res = {
        name
        for name, patts in PATTS.items()
        if any(mol.HasSubstructMatch(patt) for patt in patts)
    }
    if "Flavanonol" in res:
        res.add("Dihydroflavonol")
        # Flavanonols contain subgraphs that can satisfy very loose
        # dihydrochalcone/flavan-3-ol SMARTS, but chemically the closed
        # heterocycle + C4 carbonyl + C3 hydroxyl should be reported as the
        # more specific flavanonol/dihydroflavonol scaffold.
        res.discard("Dihydrochalcone")
        res.discard("Flavan3ol")
        res.discard("Flavanone")
    _cache[smi] = res
    return res


def is_flavonoid(smiles: str) -> bool:
    """Quick check: is this a flavonoid?"""
    return bool(classify(smiles) & CORE_FLAVONOID_CLASSES)


def keep_by_classes(df, classes):
    """df must contain a CanonicalSMILES column; classes is the list of allowed subclasses."""
    if df is None or len(df) == 0:
        return df
    if not classes:
        return df
    allowed = set(classes)
    keep = [i for i in df.index if classify(str(df.loc[i, "CanonicalSMILES"])) & allowed]
    return df.loc[keep].reset_index(drop=True) if keep else df.iloc[0:0].copy()
