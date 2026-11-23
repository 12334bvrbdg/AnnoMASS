# -*- coding: utf-8 -*-

import re
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

from core.annotating.candidates import search_from_database, search_from_pubchem, neutral_mass_from_spectrum
from core.annotating.formula import calc_formula_score
from core.annotating.structure import calc_deepmass_score, calc_matchms_score
from core.flavclass_filter import CORE_FLAVONOID_CLASSES, classify as strict_flavonoid_classify

DEBUG_COVERAGE = False
DEFAULT_ONLY_FLAVONOIDS = False
DEFAULT_ALLOWED_ELEMENTS = ""

# ── DreaMS flavonoid-library mass-window gate (2026-09-16) ───────────────
# `set_flavonoid_flags()` previously only checked the structure candidate DB
# (Stru/FlavonoidStructureDB-dedup.csv, ~18k compounds) for a flavonoid-
# classified mass match -- but that DB and DreaMS's own reference library
# are independently built and don't fully overlap, so a genuine flavonoid
# absent from the structure DB at that exact mass could get `is_flavonoid`
# =False and be excluded from the DreaMS route entirely, even when DreaMS's
# own (much larger) reference library actually has a matching entry. Fixed
# by additionally checking a mass-sorted index of
# data/reference_libraries/MassSpecGym_DreaMS_flavonoid_only.hdf5 (the same
# flavonoid-scoped library DreaMS itself now searches, see
# core/dreams_adapter.py) -- any match there is already flavonoid-relevant
# by construction (SMARTS-selected), no extra classification needed.
_DREAMS_FLAVLIB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "reference_libraries" / "MassSpecGym_DreaMS_flavonoid_only.hdf5"
)
_dreams_flavlib_sorted_masses = None  # lazy-loaded, cached module-level


def _load_dreams_flavlib_masses():
    global _dreams_flavlib_sorted_masses
    if _dreams_flavlib_sorted_masses is not None:
        return _dreams_flavlib_sorted_masses
    try:
        import h5py
        with h5py.File(_DREAMS_FLAVLIB_PATH, "r") as f:
            masses = np.asarray(f["PARENT_MASS"][:], dtype=float)
        masses = masses[np.isfinite(masses)]
        _dreams_flavlib_sorted_masses = np.sort(masses)
    except Exception:
        _dreams_flavlib_sorted_masses = np.asarray([], dtype=float)
    return _dreams_flavlib_sorted_masses


def _dreams_flavlib_mass_hit(neutral_mass, ppm=50.0):
    """True if DreaMS's own flavonoid-scoped reference library has any
    entry within `ppm` of `neutral_mass` (a cheap mass-window lookup, no
    DreaMS embedding computation)."""
    if neutral_mass is None:
        return False
    try:
        neutral_mass = float(neutral_mass)
    except (TypeError, ValueError):
        return False
    if not neutral_mass or neutral_mass <= 0:
        return False
    sorted_masses = _load_dreams_flavlib_masses()
    if sorted_masses.size == 0:
        return False
    window = neutral_mass * ppm / 1e6
    lo = bisect_left(sorted_masses, neutral_mass - window)
    hi = bisect_right(sorted_masses, neutral_mass + window)
    return hi > lo

# FIDDLE default configuration
DEFAULT_USE_FIDDLE = False
DEFAULT_FIDDLE_INSTRUMENT = "Orbitrap"
DEFAULT_FIDDLE_ELEMENTS = ""   # empty=no filter; "CHO"=keep only C/H/O formulas

def score_get(score_obj, key, default=0.0):
    if score_obj is None:
        return float(default)
    try:
        if hasattr(score_obj, "get"):
            v = score_obj.get(key, default)
        else:
            v = score_obj[key]
    except Exception:
        return float(default)
    if v is None:
        return float(default)
    try:
        if isinstance(v, float) and np.isnan(v):
            return float(default)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return float(default)

_formula_elem_re = re.compile(r"([A-Z][a-z]?)\d*")

def formula_elements(formula):
    if formula is None:
        return set()
    s = str(formula).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return set()
    return set(_formula_elem_re.findall(s))

def allowed_set_from_str(s):
    if s is None:
        return set()
    s2 = str(s).strip()
    if s2 == "":
        return set()
    return set(_formula_elem_re.findall(s2))

_BROAD_PATTS = None
_CLASS_PATTS = None
_CLASS_CACHE = {}

def _get_broad_patterns():
    global _BROAD_PATTS
    if _BROAD_PATTS is not None:
        return _BROAD_PATTS
    from rdkit import Chem
    smarts_list = [
        # ---- non-aromatic form ----
        # Flavone
        "O=C1C=C(-c2ccccc2)Oc2ccccc21",
        # generic chromone core
        "O=C1C=COc2ccccc12",
        # Flavanone
        "O=C1CCOc2ccccc12",
        "O=C1Cc2ccccc2OC1-c1ccccc1",
        # Isoflavone
        "O=C1C(=COc2ccccc21)c1ccccc1",
        # Chalcone
        "O=C(C=Cc1ccccc1)c1ccccc1",
        "c1ccccc1C(=O)C=Cc1ccccc1",
        # Aurone
        "O=C1OC(=Cc2ccccc2)c2ccccc12",
        # Anthocyanidin
        "c1cc2occc([O+]=c2cc1)c1ccccc1",
        "[O+]1=c2c(cccc2)oc2ccccc12",
        # Flavan
        "C1CCOc2ccccc12",
        # Dihydroflavonol
        "O=C1C(O)Cc2ccccc2O1",
        "OC1C(=O)c2ccccc2OC1-c1ccccc1",
        # Isoflavan
        "C1Cc2ccccc2OCC1-c1ccccc1",
        # Pterocarpan
        "C1OC2=CC=CC=C2C2OC3=CC=CC=C3C12",

        # ---- aromatic form (Daylight/RDKit) ----
        # Flavone aromatic
        "O=c1cc(-c2ccccc2)oc2ccccc12",
        "O=c1c(-c2ccccc2)coc2ccccc12",
        # chromone aromatic
        "O=c1ccoc2ccccc12",
        # Flavonol aromatic
        "O=c1c(O)c(-c2ccccc2)oc2ccccc12",
        # Isoflavone aromatic
        "O=c1oc2ccccc2c(-c2ccccc2)c1",
        "O=c1oc2ccccc2cc1-c1ccccc1",
        # quercetin and other polyhydroxyflavone aromatic
        "O=c1c(O)c(-c2ccccc2)oc2cccc(O)c12",
        "O=c1c(O)c(-c2ccccc2)oc2cc(O)cc(O)c12",
        # generic aromatic chromone core
        "O=c1coc2ccccc2c1",
        "O=c1ccoc2ccccc21",
    ]
    patts = []
    for sm in smarts_list:
        try:
            pt = Chem.MolFromSmarts(sm)
            if pt is not None:
                patts.append(pt)
        except Exception:
            pass
    _BROAD_PATTS = patts
    return _BROAD_PATTS

def _get_class_patterns():
    global _CLASS_PATTS
    if _CLASS_PATTS is not None:
        return _CLASS_PATTS
    from rdkit import Chem

    # Each class maps to multiple SMARTS (non-aromatic + aromatic forms)
    d = {
        "Flavone": [
            "O=C1C=C(-c2ccccc2)Oc2ccccc21",     # non-aromatic
            "O=c1cc(-c2ccccc2)oc2ccccc12",        # aromatic
            "O=c1c(-c2ccccc2)coc2ccccc12",        # aromatic variant
        ],
        "Flavonol": [
            "O=C1C(O)=C(-c2ccccc2)Oc2ccccc21",   # non-aromatic
            "O=c1c(O)c(-c2ccccc2)oc2ccccc12",     # aromatic
            "O=c1c(O)c(-c2ccccc2)oc2cc(O)cc(O)c12", # polyhydroxy aromatic
        ],
        "Flavanone": [
            "O=C1Cc2ccccc2OC1-c1ccccc1",          # non-aromatic
            "O=C1CC(c2ccccc2)Oc2ccccc21",         # non-aromatic variant
            "O=C1Cc2cc(O)cc(O)c2OC1c1ccc(O)cc1", # naringenin-type
        ],
        "Flavan": [
            "C1CCOc2ccccc12",
        ],
        "Isoflavone": [
            "O=C1C(=COc2ccccc21)c1ccccc1",        # non-aromatic
            "O=c1oc2ccccc2c(-c2ccccc2)c1",        # aromatic
            "O=c1oc2ccccc2cc1-c1ccccc1",          # aromatic variant
            "O=c1c(-c2ccccc2)coc2ccccc12",        # genistein-type
        ],
        "Chalcone": [
            "O=C(C=Cc1ccccc1)c1ccccc1",
            "c1ccccc1C(=O)C=Cc1ccccc1",
        ],
        "Aurone": [
            "O=C1OC(=Cc2ccccc2)c2ccccc12",
        ],
        "Anthocyanidin": [
            "c1cc2occc([O+]=c2cc1)c1ccccc1",
            "[O+]1=c2c(cccc2)oc2ccccc12",
        ],
        "Dihydroflavonol": [
            "OC1C(=O)c2ccccc2OC1-c1ccccc1",
            "O=C1C(O)Cc2ccccc2O1",
        ],
        "Isoflavan": [
            "C1Cc2ccccc2OCC1-c1ccccc1",
        ],
        "Pterocarpan": [
            "C1OC2=CC=CC=C2C2OC3=CC=CC=C3C12",
        ],
    }

    patts = {}
    for name, smarts_list in d.items():
        plist = []
        for sm in smarts_list:
            try:
                pt = Chem.MolFromSmarts(sm)
                if pt is not None:
                    plist.append(pt)
            except Exception:
                pass
        if plist:
            patts[name] = plist
    _CLASS_PATTS = patts
    return _CLASS_PATTS

def is_broad_flavonoid(smiles):
    # Compatibility name kept for older callers. The GUI/Pipeline should not
    # treat generic chromone/xanthone-like scaffolds as true flavonoids, so the
    # final yes/no gate uses the stricter subclass classifier.
    if smiles is None:
        return False
    smi = str(smiles).strip()
    if smi == "" or smi.lower() in ("nan", "none", "null"):
        return False
    try:
        return bool(set(strict_flavonoid_classify(smi)) & CORE_FLAVONOID_CLASSES)
    except Exception:
        return False

def classify_classes(smiles):
    if smiles is None:
        return set()
    smi = str(smiles).strip()
    if smi == "" or smi.lower() in ("nan", "none", "null"):
        return set()
    if smi in _CLASS_CACHE:
        return _CLASS_CACHE[smi]
    try:
        res = set(strict_flavonoid_classify(smi)) & CORE_FLAVONOID_CLASSES
        _CLASS_CACHE[smi] = res
        return res
    except Exception:
        _CLASS_CACHE[smi] = set()
        return set()

def filter_candidates(df, only_flavonoids, allowed_elements, flav_classes):
    if df is None or len(df) == 0:
        return df
    out = df

    allowed = allowed_set_from_str(allowed_elements)
    if allowed:
        keep = []
        for i in out.index:
            elems = formula_elements(out.loc[i, "MolecularFormula"])
            if elems and elems.issubset(allowed):
                keep.append(i)
        out = out.loc[keep].reset_index(drop=True) if keep else out.iloc[0:0].copy()

    if only_flavonoids:
        keep = []
        for i in out.index:
            if classify_classes(out.loc[i, "CanonicalSMILES"]):
                keep.append(i)
        out = out.loc[keep].reset_index(drop=True) if keep else out.iloc[0:0].copy()

    if flav_classes:
        allowed_cls = set([str(x).strip() for x in flav_classes if str(x).strip() != ""])
        if allowed_cls:
            keep = []
            for i in out.index:
                cls = classify_classes(out.loc[i, "CanonicalSMILES"])
                if cls & allowed_cls:
                    keep.append(i)
            out = out.loc[keep].reset_index(drop=True) if keep else out.iloc[0:0].copy()

    return out

def set_flavonoid_flags(s, cand_df):
    try:
        if cand_df is None or len(cand_df) == 0:
            s.set("is_flavonoid", False)
            s.set("flav_candidate_count", 0)
            s.set("flav_candidate_classes", [])
            return
        flav_cnt = 0
        classes_hit = set()
        for i in cand_df.index:
            smi = cand_df.loc[i, "CanonicalSMILES"]
            cls = classify_classes(smi)
            if cls:
                flav_cnt += 1
                classes_hit |= cls
        s.set("is_flavonoid", flav_cnt > 0)
        s.set("flav_candidate_count", int(flav_cnt))
        s.set("flav_candidate_classes", sorted(list(classes_hit)))
    except Exception:
        s.set("is_flavonoid", False)
        s.set("flav_candidate_count", 0)
        s.set("flav_candidate_classes", [])

def identify_unknown(s, p, model, references, database, candidate_ppm=50):
    try:
        s.set("is_flavonoid", False)
        s.set("flav_candidate_count", 0)
        s.set("flav_candidate_classes", [])

        only_flav = s.get("only_flavonoids")
        only_flav = DEFAULT_ONLY_FLAVONOIDS if only_flav is None else bool(only_flav)

        allowed_elements = s.get("allowed_elements")
        allowed_elements = DEFAULT_ALLOWED_ELEMENTS if allowed_elements is None else str(allowed_elements).strip().upper()

        flav_classes = s.get("flav_classes") or []
        if not isinstance(flav_classes, (list, tuple)):
            flav_classes = []

        # ── FIDDLE molecular formula prediction ─────────────────
        use_fiddle = s.get("use_fiddle")
        use_fiddle = DEFAULT_USE_FIDDLE if use_fiddle is None else bool(use_fiddle)

        fiddle_formulas = None   # None 表示不限制；[] 表示预测但无结果
        if use_fiddle:
            try:
                from core.fiddle_adapter import predict_formulas as fiddle_predict, is_available as fiddle_ok
                fiddle_instrument = s.get("fiddle_instrument") or DEFAULT_FIDDLE_INSTRUMENT
                fiddle_elements   = s.get("fiddle_elements")
                if fiddle_elements is None:
                    fiddle_elements = DEFAULT_FIDDLE_ELEMENTS
                if fiddle_ok(fiddle_instrument):
                    fiddle_formulas, fiddle_details = fiddle_predict(
                        s,
                        instrument=fiddle_instrument,
                        allowed_elements=fiddle_elements,
                        top_k=5,
                    )
                    s.set("fiddle_formulas", fiddle_formulas)
                    s.set("fiddle_details", fiddle_details)   # detailed prediction info
                else:
                    print(f"[FIDDLE] Model unavailable (instrument={fiddle_instrument}), skipping")
            except Exception as _fe:
                print(f"[FIDDLE] Prediction error: {_fe}")
        # ────────────────────────────────────────────────────────

        try:
            ppm = float(s.get("candidate_ppm") or candidate_ppm)
        except Exception:
            ppm = 50

        # Recorded unconditionally (search_from_database only computes this
        # itself when no formula is already set) so the MetGenX de novo
        # fallback (core/annomass_algorithm.py) always has it available, even
        # when the database candidate search below finds nothing.
        try:
            s.set("neutral_mass", neutral_mass_from_spectrum(s))
        except Exception:
            s.set("neutral_mass", None)

        # DreaMS flavonoid-library mass gate: checked independently of the
        # structure-DB search below so a flavonoid absent from the structure
        # DB (which would otherwise leave is_flavonoid False and exclude the
        # spectrum from the DreaMS route entirely) still gets a chance when
        # DreaMS's own reference library actually covers it.
        _dreams_flav_hit = _dreams_flavlib_mass_hit(s.get("neutral_mass"), ppm=ppm)
        if _dreams_flav_hit:
            s.set("is_flavonoid", True)

        s = search_from_database(s, database, ppm=ppm)
        cand = s.get("annotation")

        if cand is None:
            # PubChem fallback disabled, using local database only
            return s

        if cand is None or len(cand) == 0:
            return s

        set_flavonoid_flags(s, cand)
        if _dreams_flav_hit:
            s.set("is_flavonoid", True)

        cand = filter_candidates(cand, only_flavonoids=only_flav, allowed_elements=allowed_elements, flav_classes=flav_classes)
        if cand is None or len(cand) == 0:
            s.set("annotation", None)
            return s

        # ── Filter candidates by FIDDLE-predicted molecular formula ──
        if fiddle_formulas is not None and len(fiddle_formulas) > 0:
            mask = cand["MolecularFormula"].isin(set(fiddle_formulas))
            filtered = cand[mask].reset_index(drop=True)
            if len(filtered) > 0:
                cand = filtered
                print(f"[FIDDLE] After filtering: {len(cand)} candidates retained ({mask.sum()} formula matches)")
            else:
                print("[FIDDLE] No candidates match predicted formulas, retaining all candidates")
        # ────────────────────────────────────────────────────────────

        s.set("annotation", cand)

        formula_score = calc_formula_score(s)
        structure_score, structure_score_raw, reference_spectrum = calc_deepmass_score(s, p, model, references)
        if structure_score is None:
            return s

        for i in cand.index:
            k = cand.loc[i, "InChIKey"]
            f = cand.loc[i, "MolecularFormula"]
            fs = score_get(formula_score, f, 0.0)
            ss = score_get(structure_score, k, 0.0)
            ssr = score_get(structure_score_raw, k, 0.0)
            cand.loc[i, "Formula Score"] = fs
            cand.loc[i, "Structure Score"] = ss
            cand.loc[i, "Consensus Score"] = 0.3 * fs + 0.7 * ss
            cand.loc[i, "DeepMASS_raw"] = ssr

        cand = cand.sort_values("Consensus Score", ignore_index=True, ascending=False)
        s.set("annotation", cand)
        s.set("reference", reference_spectrum)
        return s

    except Exception as e:
        title = s.metadata.get("compound_name", "Unknown")
        msg = str(e)
        # MS1-only entries (MS2SCAN=-1 / no fragments) follow the original
        # failure path below; suppress only the noisy terminal message.
        if "zero-size array to reduction operation maximum" not in msg:
            print(title, "error:", e)
        s.set("annotation", None)
        return s

def match_spectrum(s, precursors, references, database, candidate_ppm=50):
    s.set("is_flavonoid", False)
    s.set("flav_candidate_count", 0)
    s.set("flav_candidate_classes", [])

    only_flav = s.get("only_flavonoids")
    only_flav = DEFAULT_ONLY_FLAVONOIDS if only_flav is None else bool(only_flav)

    allowed_elements = s.get("allowed_elements")
    allowed_elements = DEFAULT_ALLOWED_ELEMENTS if allowed_elements is None else str(allowed_elements).strip().upper()

    flav_classes = s.get("flav_classes") or []
    if not isinstance(flav_classes, (list, tuple)):
        flav_classes = []

    try:
        ppm = float(s.get("candidate_ppm") or candidate_ppm)
    except Exception:
        ppm = 50
    s = search_from_database(s, database, ppm=ppm)
    cand = s.get("annotation")

    if cand is None:
        # PubChem fallback disabled, using local database only
        return s

    if cand is None or len(cand) == 0:
        return s

    set_flavonoid_flags(s, cand)

    cand = filter_candidates(cand, only_flavonoids=only_flav, allowed_elements=allowed_elements, flav_classes=flav_classes)
    if cand is None or len(cand) == 0:
        s.set("annotation", None)
        return s
    s.set("annotation", cand)

    formula_score = calc_formula_score(s)
    structure_score, reference_spectrum = calc_matchms_score(s, precursors, references)
    if structure_score is None:
        return s

    for i in cand.index:
        k = cand.loc[i, "InChIKey"]
        f = cand.loc[i, "MolecularFormula"]
        fs = score_get(formula_score, f, 0.0)
        ss = score_get(structure_score, k, 0.0)
        cand.loc[i, "Formula Score"] = fs
        cand.loc[i, "Structure Score"] = ss
        cand.loc[i, "Consensus Score"] = 0.3 * fs + 0.7 * ss

    cand = cand.sort_values("Consensus Score", ignore_index=True, ascending=False)
    s.set("annotation", cand)
    s.set("reference", reference_spectrum)
    return s


_DB_CACHE_BUILT = False

def prebuild_database_cache(database):
    global _DB_CACHE_BUILT
    if _DB_CACHE_BUILT:
        return
    if database is None or len(database) == 0:
        _DB_CACHE_BUILT = True
        return
    col = "CanonicalSMILES"
    if col not in database.columns:
        _DB_CACHE_BUILT = True
        return
    smiles_list = database[col].dropna().unique().tolist()
    print("Prebuilding flavonoid cache for {} SMILES...".format(len(smiles_list)))
    for smi in smiles_list:
        smi = str(smi).strip()
        if smi == "" or smi.lower() in ("nan", "none", "null"):
            continue
        is_broad_flavonoid(smi)
        classify_classes(smi)
    _DB_CACHE_BUILT = True
    print("Cache built: {} entries".format(len(_CLASS_CACHE)))
