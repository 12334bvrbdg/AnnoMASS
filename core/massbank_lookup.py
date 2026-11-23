"""Live MassBank lookup for the Flavonoid Agent, via the public MassBank3
REST API (https://massbank.eu/MassBank-api) -- confirmed live and working
directly against the real endpoint before this module was written. Covers
compounds beyond the project's local 154-compound curated MassBank
standard-reference subset (core/flavonoid_reference_db.py). Read-only,
network-dependent, never used to filter/rank Pipeline candidates.
"""

import re
from typing import Any, Dict, List, Optional

import requests

API_BASE = "https://massbank.eu/MassBank-api"
TIMEOUT = 15

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}(-[A-Z]{10}-[A-Z])?$")


def _sniff_field(q: str) -> str:
    q = q.strip()
    if _INCHIKEY_RE.match(q.upper()):
        return "inchikey"
    return "name"


def _get_subtag(subtags: List[Dict[str, Any]], name: str) -> Optional[str]:
    for s in subtags or []:
        if s.get("subtag") == name:
            return s.get("value")
    return None


def _summarize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    compound = rec.get("compound", {}) or {}
    names = compound.get("names") or []
    inchikey = None
    for link in compound.get("link") or []:
        if link.get("database") == "INCHIKEY":
            inchikey = link.get("identifier")
            break

    acquisition = rec.get("acquisition", {}) or {}
    ms = acquisition.get("mass_spectrometry", {}) or {}
    focused_ion = (rec.get("mass_spectrometry", {}) or {}).get("focused_ion") or []
    chromatography = acquisition.get("chromatography") or []
    peak = rec.get("peak", {}) or {}

    return {
        "accession": rec.get("accession"),
        "compound_name": names[0] if names else None,
        "formula": compound.get("formula"),
        "exact_mass": compound.get("mass"),
        "smiles": compound.get("smiles"),
        "inchikey": inchikey,
        "ion_mode": ms.get("ion_mode"),
        "ms_type": ms.get("ms_type"),
        "precursor_mz": _get_subtag(focused_ion, "PRECURSOR_M/Z"),
        "precursor_type": _get_subtag(focused_ion, "PRECURSOR_TYPE"),
        "instrument_type": acquisition.get("instrument_type"),
        "retention_time": _get_subtag(chromatography, "RETENTION_TIME"),
        "num_peaks": peak.get("numPeak"),
        "splash": peak.get("splash"),
        "massbank_url": f"https://massbank.eu/MassBank/RecordDisplay?id={rec.get('accession')}" if rec.get("accession") else None,
    }


def query_massbank(query: str, field: str = "auto", limit: int = 10) -> Dict[str, Any]:
    """field: 'auto' | 'name' | 'inchikey'. Returns a compact per-record
    summary (NOT the full peak list -- keep tool output small; a broad name
    query can match hundreds of records) plus a total_matches count."""
    if not query or not query.strip():
        return {"error": "Empty query."}
    resolved = _sniff_field(query) if field == "auto" else field
    if resolved not in ("name", "inchikey"):
        return {"error": f"Unsupported field '{resolved}' for MassBank lookup (use name/inchikey)."}

    params = {"compound_name": query.strip()} if resolved == "name" else {"inchi_key": query.strip()}
    try:
        resp = requests.get(f"{API_BASE}/records", params=params, timeout=TIMEOUT,
                             headers={"User-Agent": "AnnoMASS-Agent/1.0"})
    except requests.RequestException as exc:
        return {"error": f"MassBank request failed: {exc}"}
    if resp.status_code == 404:
        return {"query": query, "total_matches": 0, "hits": []}
    if resp.status_code != 200:
        return {"error": f"MassBank returned HTTP {resp.status_code}: {resp.text[:300]}"}
    try:
        records = resp.json()
    except ValueError as exc:
        return {"error": f"Unexpected MassBank response shape: {exc}"}
    if not isinstance(records, list):
        return {"error": "Unexpected MassBank response shape (expected a list)."}

    hits = [_summarize_record(r) for r in records[:limit]]
    return {"query": query, "total_matches": len(records), "hits": hits}
