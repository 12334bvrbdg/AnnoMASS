"""Live PubChem lookup for the Flavonoid Agent -- for compounds not found in
the local structure DB (core/structure_db_query.py) or reference spectra
(core/flavonoid_reference_db.py). Read-only, network-dependent, never used
to filter/rank Pipeline candidates.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# NOTE: PubChem deprecated the "CanonicalSMILES"/"IsomericSMILES" property
# names (silently dropped from the response, no error) -- "SMILES" is the
# current name for isomeric SMILES.
PROPERTIES = "MolecularFormula,SMILES,IUPACName,MolecularWeight,InChIKey"
TIMEOUT = 15

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}(-[A-Z]{10}-[A-Z])?$")


def _sniff_field(q: str) -> str:
    q = q.strip()
    if _INCHIKEY_RE.match(q.upper()):
        return "inchikey"
    if any(c in q for c in "()=#@[]"):
        try:
            from rdkit import Chem
            if Chem.MolFromSmiles(q) is not None:
                return "smiles"
        except Exception:
            pass
    return "name"


def query_pubchem(query: str, field: str = "auto", limit: int = 5) -> Dict[str, Any]:
    """field: 'auto' | 'name' | 'smiles' | 'inchikey'. Returns {"hits": [...]} or {"error": ...}."""
    if not query or not query.strip():
        return {"error": "Empty query."}
    resolved = _sniff_field(query) if field == "auto" else field
    if resolved not in ("name", "smiles", "inchikey"):
        return {"error": f"Unsupported field '{resolved}' for PubChem lookup (use name/smiles/inchikey)."}

    encoded = quote(query.strip(), safe="")
    url = f"{PUG_BASE}/compound/{resolved}/{encoded}/property/{PROPERTIES}/JSON"
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "AnnoMASS-Agent/1.0"})
    except requests.RequestException as exc:
        return {"error": f"PubChem request failed: {exc}"}
    if resp.status_code == 404:
        return {"hits": []}
    if resp.status_code != 200:
        return {"error": f"PubChem returned HTTP {resp.status_code}: {resp.text[:300]}"}
    try:
        data = resp.json()
        props = data["PropertyTable"]["Properties"]
    except (ValueError, KeyError) as exc:
        return {"error": f"Unexpected PubChem response shape: {exc}"}

    hits = []
    for p in props[:limit]:
        hits.append({
            "cid": p.get("CID"),
            "formula": p.get("MolecularFormula"),
            "smiles": p.get("SMILES"),
            "iupac_name": p.get("IUPACName"),
            "molecular_weight": p.get("MolecularWeight"),
            "inchikey": p.get("InChIKey"),
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID')}" if p.get("CID") else None,
        })
    return {"query": query, "field": resolved, "hits": hits}
