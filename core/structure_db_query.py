"""Flexible ad hoc query over the live flavonoid structure database
(main_window.default_database, loaded once at annomass_app.py:1557 from
Stru/FlavonoidStructureDB-dedup.csv -- columns: Title, Formula, SMILES,
InChIkey, Exact mass, Database IDs).

core/annotating/candidates.py already has retrieve_by_formula_database() and
retrieve_by_exact_mass_database(), but those are pipeline-internal (rename
columns, ppm-window contract specific to candidate retrieval during
identification) and have no name/InChIKey lookup path. This module is a
separate, agent-facing query surface -- it does not import or modify
anything in core/annotating/candidates.py, so it carries zero risk to the
existing pipeline contract.

Read-only: never used to filter/rank Pipeline candidates, only to answer
ad hoc "what do we know about compound X" questions.
"""

import re
from typing import Any, Dict, List

import pandas as pd

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_INCHIKEY_PREFIX_RE = re.compile(r"^[A-Z]{14}$")
_FORMULA_RE = re.compile(r"^[A-Za-z0-9]+$")


def _sniff_field(query: str) -> str:
    q = query.strip()
    if _INCHIKEY_RE.match(q) or _INCHIKEY_PREFIX_RE.match(q):
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


def query_structure_db(database: pd.DataFrame, query: str, field: str = "auto", limit: int = 20) -> List[Dict[str, Any]]:
    """Search the live structure DB. field: "auto" | "name" | "inchikey" | "formula" | "smiles"."""
    if database is None or len(database) == 0 or not query or not query.strip():
        return []
    q = query.strip()
    resolved_field = _sniff_field(q) if field == "auto" else field

    if resolved_field == "inchikey":
        mask = database["InChIkey"].astype(str).str.upper().str.startswith(q.upper())
    elif resolved_field == "formula":
        mask = database["Formula"].astype(str).str.upper() == q.upper()
    elif resolved_field == "smiles":
        canonical = _canonical_smiles(q)
        if canonical is None:
            mask = database["SMILES"].astype(str) == q
        else:
            mask = database["SMILES"].astype(str).map(_canonical_smiles) == canonical
    else:  # name substring
        mask = database["Title"].astype(str).str.contains(re.escape(q), case=False, na=False)

    hits = database[mask].head(limit)
    return hits.to_dict(orient="records")


def _canonical_smiles(smiles: str):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None
