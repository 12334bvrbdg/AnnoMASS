#!/usr/bin/env python3
"""Screen flavonoids from any structure table with the rules used to build the AnnoMASS
flavonoid structure library (FlavonoidStructureDB-dedup.csv).

A structure is kept when
  1. it contains only C, H and O (largest fragment; counter-ions are ignored),
  2. it has at least 15 carbon atoms (the C6-C3-C6 skeleton),
  3. it contains a flavonoid core: a match to one of the core SMARTS below, or a flavonoid
     subclass recognized by core/flavclass_filter.py (flavone, flavonol, flavanone,
     flavanonol, isoflavone, isoflavanone, isoflavan, chalcone, dihydrochalcone, aurone,
     anthocyanidin, flavan-3-ol, flavan, pterocarpan, ...).
Duplicates are removed by InChIKey (first occurrence kept).

Input: CSV, TSV or TXT, optionally gzip-compressed. The SMILES column is found by name
(any column containing "smiles"); use --smiles-col to set it. A file without a header
(for example PubChem CID-SMILES.gz: "CID<TAB>SMILES") is also accepted with --no-header.

Examples
  python flavonoid_screening/screen_flavonoids.py --input DeepMassStructureDB-v1.1.csv --out flavonoids.csv
  python flavonoid_screening/screen_flavonoids.py --input CID-SMILES.gz --no-header \\
         --id-col 0 --smiles-col 1 --out pubchem_flavonoids.csv --workers 8
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.flavclass_filter import is_flavonoid as _flavclass_is_flavonoid  # noqa: E402

# Flavonoid core SMARTS (flavone, flavonol, flavanone, flavan, isoflavone, chalcone, aurone,
# flavylium, flavanonol, isoflavan, pterocarpan and chromone cores)
FLAVONOID_SMARTS = [
    "O=C1C=C(-c2ccccc2)Oc2ccccc21",
    "O=c1cc(-c2ccccc2)oc2ccccc12",
    "O=c1c(-c2ccccc2)coc2ccccc12",
    "O=C1C(O)=C(-c2ccccc2)Oc2ccccc21",
    "O=c1c(O)c(-c2ccccc2)oc2ccccc12",
    "O=c1c(O)c(-c2ccccc2)oc2cc(O)cc(O)c12",
    "O=C1Cc2ccccc2OC1-c1ccccc1",
    "O=C1CC(c2ccccc2)Oc2ccccc21",
    "O=C1Cc2cc(O)cc(O)c2OC1c1ccc(O)cc1",
    "C1CCOc2ccccc12",
    "O=C1C(=COc2ccccc21)c1ccccc1",
    "O=c1oc2ccccc2c(-c2ccccc2)c1",
    "O=c1oc2ccccc2cc1-c1ccccc1",
    "O=C(C=Cc1ccccc1)c1ccccc1",
    "c1ccccc1C(=O)C=Cc1ccccc1",
    "O=C1OC(=Cc2ccccc2)c2ccccc12",
    "c1cc2occc([O+]=c2cc1)c1ccccc1",
    "[O+]1=c2c(cccc2)oc2ccccc12",
    "OC1C(=O)c2ccccc2OC1-c1ccccc1",
    "O=C1C(O)Cc2ccccc2O1",
    "C1Cc2ccccc2OCC1-c1ccccc1",
    "C1OC2=CC=CC=C2C2OC3=CC=CC=C3C12",
    "O=C1C=COc2ccccc12",
    "O=c1ccoc2ccccc12",
    "O=c1coc2ccccc2c1",
]
_PATTERNS = [p for p in (Chem.MolFromSmarts(s) for s in FLAVONOID_SMARTS) if p is not None]
# fast pre-filter: SMILES with any element other than C, H, O (N, S, P, halogens, metals ...)
_NON_CHO = re.compile(r"[ABD-GI-NP-Zabd-gi-np-z]")
_CHOOSER = rdMolStandardize.LargestFragmentChooser()
MIN_CARBONS = 15


def screen_one(smiles: str):
    """Return (InChIKey, formula, exact mass, canonical SMILES) for a flavonoid, else None."""
    if not smiles or ("." not in smiles and _NON_CHO.search(smiles)):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if "." in smiles:
        mol = _CHOOSER.choose(mol)
    if {a.GetSymbol() for a in mol.GetAtoms()} - {"C", "O", "H"}:
        return None
    if sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C") < MIN_CARBONS:
        return None
    can = Chem.MolToSmiles(mol)
    if not (any(mol.HasSubstructMatch(p) for p in _PATTERNS) or _flavclass_is_flavonoid(can)):
        return None
    key = inchi.MolToInchiKey(mol)
    if not key:
        return None
    return key, CalcMolFormula(mol), round(Descriptors.ExactMolWt(mol), 6), can


def _open(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace", newline="")


def read_rows(args):
    """Yield (row id, name, SMILES, original row dict or None)."""
    with _open(args.input) as fh:
        first = fh.readline()
        delim = "\t" if first.count("\t") >= first.count(",") else ","
        fh_rows = csv.reader([first], delimiter=delim)
        if args.no_header:
            sc, ic = int(args.smiles_col or 1), int(args.id_col if args.id_col is not None else 0)
            for i, row in enumerate(_chain(fh_rows, csv.reader(fh, delimiter=delim))):
                if len(row) > sc:
                    yield (row[ic] if len(row) > ic else str(i)), "", row[sc], None
            return
        header = next(fh_rows)
        reader = csv.DictReader(fh, fieldnames=header, delimiter=delim)
        sc = args.smiles_col or next((h for h in header if "smiles" in h.lower()), None)
        if sc is None:
            sys.exit(f"No SMILES column found in {header}; use --smiles-col")
        nc = next((h for h in header if h.lower() in ("title", "name", "compound_name")), None)
        ic = args.id_col or next((h for h in header if h.lower() in ("id", "cid", "database ids")), None)
        for i, row in enumerate(reader):
            yield (row.get(ic, "") if ic else str(i)), (row.get(nc, "") if nc else ""), row.get(sc, ""), row


def _chain(*its):
    for it in its:
        yield from it


def _work(item):
    rid, name, smi, row = item
    res = screen_one(smi)
    return None if res is None else (rid, name, row, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="structure table (.csv/.tsv/.txt, optionally .gz)")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--smiles-col", help="SMILES column name (or 0-based index with --no-header)")
    ap.add_argument("--id-col", help="ID column name (or 0-based index with --no-header)")
    ap.add_argument("--no-header", action="store_true", help="input has no header line")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    args = ap.parse_args()

    seen, n_in, n_out = set(), 0, 0
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["Title", "Formula", "SMILES", "InChIkey", "Exact mass", "Database IDs"])
        rows = read_rows(args)
        results = (Pool(args.workers).imap(_work, rows, chunksize=2000) if args.workers > 1
                   else map(_work, rows))
        for res in results:
            n_in += 1
            if n_in % 1_000_000 == 0:
                print(f"  {n_in:,} structures read, {n_out:,} flavonoids", file=sys.stderr)
            if res is None:
                continue
            rid, name, row, (key, formula, mass, can) = res
            if key in seen:
                continue
            seen.add(key)
            w.writerow([name, formula, can, key, mass, rid])
            n_out += 1
    print(f"{n_in:,} structures read, {n_out:,} unique flavonoids written to {args.out}")


if __name__ == "__main__":
    main()
