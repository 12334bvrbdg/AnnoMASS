"""Lazy loader/query for the static flavonoid_canonical_dataset/ folder.

There is no runtime loader for this dataset anywhere else in the project --
it's purely static data on disk (4 sources, each an MGF + CSV pair; see
flavonoid_canonical_dataset/README.md). This module answers "does compound X
have a reference spectrum, and from which source(s)" for the Flavonoid
Agent's query_reference_spectra tool.

Read-only, static-data lookup -- never used to filter/rank Pipeline
candidates, only to answer ad hoc reference-library questions.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DATASET_ROOT = Path(__file__).resolve().parent.parent / "flavonoid_canonical_dataset"

MASSBANK_CSV = DATASET_ROOT / "standard_reference_only" / "standard_reference_spectra_full_inchikey.csv"
OTHER_SOURCES = {
    "MassSpecGym": DATASET_ROOT / "other_sources" / "processed" / "MassSpecGym" / "massspecgym_flavonoid_clean_compounds.csv",
    "GNPS": DATASET_ROOT / "other_sources" / "processed" / "GNPS" / "gnps_selected_flavonoid_clean_compounds.csv",
    "MoNA": DATASET_ROOT / "other_sources" / "processed" / "MoNA" / "mona_lcmsms_flavonoid_clean_compounds.csv",
}

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}(-[A-Z]{10}-[A-Z])?$")
_FORMULA_RE = re.compile(r"^[A-Za-z0-9]+$")

_CACHE: Dict[str, pd.DataFrame] = {}


def _load_massbank() -> pd.DataFrame:
    if not MASSBANK_CSV.exists():
        return pd.DataFrame(columns=["source", "compound_name", "full_inchikey", "inchikey14",
                                      "canonical_smiles", "formula", "n_spectra", "adducts", "flavonoid_classes"])
    raw = pd.read_csv(MASSBANK_CSV, low_memory=False)
    rows = []
    for inchikey, group in raw.groupby("inchikey"):
        first = group.iloc[0]
        rows.append({
            "source": "MassBank",
            "compound_name": str(first.get("names", "") or "").split("|")[0],
            "full_inchikey": inchikey,
            "inchikey14": first.get("inchikey14", ""),
            "canonical_smiles": first.get("canonical_smiles", ""),
            "formula": first.get("rdkit_formula", first.get("formula", "")),
            "n_spectra": len(group),
            "adducts": ";".join(sorted(set(str(x) for x in group["precursor_type"].dropna().unique()))),
            "flavonoid_classes": first.get("flavonoid_classes", ""),
        })
    return pd.DataFrame(rows)


def _load_compound_csv(source_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["source", "compound_name", "full_inchikey", "inchikey14",
                                      "canonical_smiles", "formula", "n_spectra", "adducts", "flavonoid_classes"])
    raw = pd.read_csv(path, low_memory=False)
    return pd.DataFrame({
        "source": source_name,
        "compound_name": "",  # these 3 sources are not name-indexed
        "full_inchikey": raw["full_inchikey"],
        "inchikey14": raw["inchikey14"],
        "canonical_smiles": raw["canonical_smiles"],
        "formula": raw["rdkit_formula"],
        "n_spectra": raw["spectra"],
        "adducts": raw["adducts"],
        "flavonoid_classes": raw.get("flavonoid_classes", ""),
    })


def _load_all() -> Dict[str, pd.DataFrame]:
    if not _CACHE:
        _CACHE["MassBank"] = _load_massbank()
        for name, path in OTHER_SOURCES.items():
            _CACHE[name] = _load_compound_csv(name, path)
    return _CACHE


def _sniff_field(q: str) -> str:
    q = q.strip()
    if _INCHIKEY_RE.match(q.upper()):
        return "inchikey"
    if _FORMULA_RE.match(q) and any(c.isupper() for c in q) and any(c.isdigit() for c in q):
        return "formula"
    if any(c in q for c in "()=#@[]"):
        try:
            from rdkit import Chem
            if Chem.MolFromSmiles(q) is not None:
                return "smiles"
        except Exception:
            pass
    return "name"


def query(name: Optional[str] = None, inchikey: Optional[str] = None, formula: Optional[str] = None,
          smiles: Optional[str] = None, source: str = "all", limit: int = 20) -> List[Dict[str, Any]]:
    """Free-form query: pass exactly one of name/inchikey/formula/smiles, or
    let a single positional-style call auto-sniff via query_reference_spectra() below."""
    tables = _load_all()
    sources = list(tables.keys()) if source in (None, "all") else [source]
    frames = [tables[s] for s in sources if s in tables]
    if not frames:
        return []
    combined = pd.concat(frames, ignore_index=True)

    if inchikey:
        ik = inchikey.strip().upper()
        mask = combined["full_inchikey"].astype(str).str.upper().str.startswith(ik) | \
               combined["inchikey14"].astype(str).str.upper().str.startswith(ik)
    elif formula:
        mask = combined["formula"].astype(str).str.upper() == formula.strip().upper()
    elif smiles:
        canonical = _canonical_smiles(smiles)
        if canonical is None:
            mask = combined["canonical_smiles"].astype(str) == smiles.strip()
        else:
            mask = combined["canonical_smiles"].astype(str).map(_canonical_smiles) == canonical
    elif name:
        mask = combined["compound_name"].astype(str).str.contains(re.escape(name.strip()), case=False, na=False)
    else:
        return []

    hits = combined[mask].head(limit)
    return hits.to_dict(orient="records")


def query_reference_spectra(query_str: str, field: str = "auto", source: str = "all", limit: int = 20) -> List[Dict[str, Any]]:
    """Single-string entry point (matches the agent tool schema): sniffs
    whether query_str looks like a name / InChIKey / formula / SMILES."""
    if not query_str or not query_str.strip():
        return []
    resolved = _sniff_field(query_str) if field == "auto" else field
    kwargs = {resolved: query_str}
    return query(source=source, limit=limit, **kwargs)


def _canonical_smiles(smiles: str):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None
