# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:51:53 2023

@author: DELL

Patched:
- Convert ion m/z (parent_mass/pepmass/precursor_mz) to neutral exact mass before
  retrieving candidates by 'Exact mass' in database / PubChem.
- Support common adducts based on ION/ADDUCT and CHARGE.
"""

import json
import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import AllChem

# ------------------ Neutral mass conversion ------------------
PROTON = 1.00727646688

# For singly-charged ions: mz = M + add_mass
# So: M = mz - add_mass
ADDUCT_MASS = {
    "[M+H]+": PROTON,
    "[M-H]-": -PROTON,

    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,

    "[M+Cl]-": 34.968853,

    "[M+HCOO]-": 44.997654,     # formate adduct
    "[M+CH3COO]-": 59.013851,   # acetate adduct

    # Some alternative notations (if your data uses them)
    "[M+FA-H]-": 44.997654,
    "[M+AC-H]-": 59.013851,
    "[M-2H]2-": -2 * PROTON,
}

def _parse_charge(charge_str):
    """
    Parse charge like '1+', '2+', '1-', '2-'.
    Return signed int.
    """
    if charge_str is None:
        return 1
    m = re.match(r"^\s*(\d)\s*([+-])\s*$", str(charge_str))
    if not m:
        return 1
    z = int(m.group(1))
    return z if m.group(2) == "+" else -z

def _get_adduct_from_spectrum(s):
    # try common keys
    adduct = s.get("adduct") or s.get("ion") or s.get("ADDUCT") or s.get("ION") or ""
    adduct = str(adduct).strip()
    if adduct == "":
        ionmode = str(s.get("ionmode") or s.get("IONMODE") or "").strip().lower()
        if ionmode.startswith("pos"):
            adduct = "[M+H]+"
        elif ionmode.startswith("neg"):
            adduct = "[M-H]-"
    return adduct

def neutral_mass_from_spectrum(s):
    """
    Compute neutral exact mass from spectrum metadata.

    We try to read an ion m/z from:
      precursor_mz > pepmass > parent_mass
    and convert to neutral mass using:
      M = mz*|z| - add_mass
    If adduct not recognized, fallback to mz*|z| (assume mz ~ neutral mass).
    """
    mz = s.get("precursor_mz") or s.get("pepmass") or s.get("parent_mass")
    if mz is None:
        return None
    try:
        mz = float(mz)
    except:
        return None

    adduct = _get_adduct_from_spectrum(s)
    charge = _parse_charge(s.get("charge") or s.get("CHARGE") or "1+")
    zabs = abs(int(charge)) if charge else 1
    if zabs == 0:
        zabs = 1

    add = ADDUCT_MASS.get(adduct, None)
    if add is None:
        # Unknown adduct: assume already neutral (or close) to not kill pipeline
        return mz * zabs

    return mz * zabs - add

# ------------------ Main search APIs ------------------
def search_from_database(s, database, ppm=10):
    # Prefer formula if provided; else use NEUTRAL mass for exact mass retrieval
    formula = s.get("formula")
    if formula is not None:
        candidates = retrieve_by_formula_database(formula, database)
    else:
        mass_neutral = neutral_mass_from_spectrum(s)
        if mass_neutral is not None:
            candidates = retrieve_by_exact_mass_database(mass_neutral, database, ppm=ppm)
        else:
            candidates = None

    if candidates is not None:
        candidates["Formula Score"] = np.nan
        candidates["Structure Score"] = np.nan
        candidates["Consensus Score"] = np.nan
        candidates["DeepMASS_raw"] = np.nan

    s.set("annotation", candidates)
    return s

def search_from_pubchem(s, ppm=10):
    formula = s.get("formula")
    if formula is not None:
        try:
            candidates = retrieve_by_formula_pubchem(formula)
        except:
            candidates = None
    else:
        mass_neutral = neutral_mass_from_spectrum(s)
        if mass_neutral is not None:
            try:
                candidates = retrieve_by_exact_mass_pubchem(mass_neutral, ppm=ppm)
            except:
                candidates = None
        else:
            candidates = None

    if candidates is not None:
        candidates["Formula Score"] = np.nan
        candidates["Structure Score"] = np.nan
        candidates["Consensus Score"] = np.nan
        candidates["DeepMASS_raw"] = np.nan

    s.set("annotation", candidates)
    return s

# ------------------ PubChem retrieval ------------------
def retrieve_by_cid_list(idlist):
    res = []
    idstring = ""
    for i, cid in enumerate(idlist):
        idstring += "," + str(cid)
        if ((i % 100 == 99) or (i == len(idlist) - 1)):
            url = (
                "http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                + idstring[1:len(idstring)]
                + "/property/MolecularFormula,InChIKey,CanonicalSMILES/JSON"
            )
            soup = BeautifulSoup(requests.get(url, timeout=15).content, "html.parser")
            properties = json.loads(str(soup))["PropertyTable"]["Properties"]
            properties = [[s[k] for k in s.keys()] for s in properties]
            res += properties
            idstring = ""
    res = pd.DataFrame(res)
    res.columns = ["CID", "MolecularFormula", "CanonicalSMILES", "InChIKey"]
    k = np.array([("." not in s) for s in res["CanonicalSMILES"]])
    res = res.loc[k, :]
    res = res.reset_index(drop=True)
    return res

# Global SMILES -> (parsed_ok, calculated_formula, canonical_smiles) cache.
# refine_compound_list runs per spectrum on mass-window hits, and the same
# candidate compounds appear again and again across spectra; RDKit parse +
# CalcMolFormula results for a given SMILES never change, so cache them.
_REFINE_CACHE = {}
_REFINE_CACHE_MAX = 500000


def _refine_entry(smi):
    entry = _REFINE_CACHE.get(smi)
    if entry is None:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            entry = (False, None, None)
        else:
            entry = (
                True,
                AllChem.CalcMolFormula(mol),
                Chem.MolToSmiles(mol, isomericSmiles=True),
            )
        if len(_REFINE_CACHE) < _REFINE_CACHE_MAX:
            _REFINE_CACHE[smi] = entry
    return entry


def refine_compound_list(res):
    keep, keys = [], []
    for i in res.index:
        smi = res.loc[i, "CanonicalSMILES"]
        formula = res.loc[i, "MolecularFormula"]
        ok, formula_cal, canonical = _refine_entry(smi)
        if not ok:
            continue
        if formula != formula_cal:
            continue
        if "." in smi:
            continue
        # Keep stereoisomers/regioisomers distinct. Using only the first
        # InChIKey block collapses compounds such as Schaftoside and
        # neoschaftoside before the strict structure intersection can decide.
        key = str(res.loc[i, "InChIKey"] or "").strip()
        if not key or key.lower() in ("nan", "none", "null"):
            key = canonical
        if key not in keys:
            keep.append(i)
            keys.append(key)
    keep, keys = np.array(keep), np.array(keys)
    return res.loc[keep]

def retrieve_by_exact_mass_pubchem(mass, ppm=10):
    min_mass = mass - mass * ppm / 10 ** 6
    max_mass = mass + mass * ppm / 10 ** 6
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pccompound&retmode=json&term={min_mass}%3A{max_mass}[ExactMass]&retmax=9999"
    )
    try:
        res = requests.get(url).text
    except:
        raise ConnectionError
    res = json.loads(res)
    idlist = res["esearchresult"]["idlist"]
    result = retrieve_by_cid_list(idlist)
    result = refine_compound_list(result)
    result = result.reset_index(drop=True)
    return result

def retrieve_by_formula_pubchem(formula, timeout=999):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{formula}/cids/json"
    try:
        res = requests.get(url).text
    except:
        raise ConnectionError
    res = json.loads(res)
    idlist = res["IdentifierList"]["CID"]
    result = retrieve_by_cid_list(idlist)
    result = refine_compound_list(result)
    result = result.reset_index(drop=True)
    return result

# ------------------ Local database retrieval ------------------
def retrieve_by_exact_mass_database(mass, database, ppm=10):
    min_mass = mass - mass * ppm / 10 ** 6
    max_mass = mass + mass * ppm / 10 ** 6

    # Ensure numeric
    # (If your CSV has Exact mass as string, convert once here)
    if database["Exact mass"].dtype == object:
        database = database.copy()
        database["Exact mass"] = pd.to_numeric(database["Exact mass"], errors="coerce")

    result = database[np.logical_and(database["Exact mass"] >= min_mass, database["Exact mass"] <= max_mass)]
    if len(result) == 0:
        return None

    # Keep original expected columns
    result = result[["Title", "Formula", "SMILES", "InChIkey", "Database IDs"]]
    result.columns = ["Title", "MolecularFormula", "CanonicalSMILES", "InChIKey", "Database IDs"]
    result = refine_compound_list(result)
    result = result.reset_index(drop=True)
    return result

def retrieve_by_formula_database(formula, database):
    result = database[database["Formula"] == formula]
    if len(result) == 0:
        return None

    result = result[["Title", "Formula", "SMILES", "InChIkey", "Database IDs"]]
    result.columns = ["Title", "MolecularFormula", "CanonicalSMILES", "InChIKey", "Database IDs"]
    result = refine_compound_list(result)
    result = result.reset_index(drop=True)
    return result

if __name__ == "__main__":
    """
    from core.importing.load_from_files import load_from_files
    database = pd.read_csv('data/DeepMassStructureDB-v1.1.csv')
    spectrums = load_from_files(["example/all_casmi.mgf"])
    spectrums = [search_from_database(s, database, ppm = 10) for s in spectrums]
    """
    pass
