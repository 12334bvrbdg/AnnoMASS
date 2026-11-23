"""MoNA (MassBank of North America) online lookup helpers.

MoNA exposes a public Angular site and several REST endpoints. The browse
endpoint is anonymous but does not reliably apply query filters; the filtered
search endpoint may require authorization on some deployments. This module
therefore tries the filtered API first, then returns actionable MoNA links and
offline-library guidance instead of pretending a blocked request succeeded.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

import requests


MONA_BASE_URL = "https://mona.fiehnlab.ucdavis.edu"
MONA_SEARCH_ENDPOINT = f"{MONA_BASE_URL}/rest/spectra/search"
MONA_BROWSER_URL = f"{MONA_BASE_URL}/#/spectra/search"
MONA_DOWNLOADS_URL = f"{MONA_BASE_URL}/#/downloads"


def query_mona(
    query: str,
    field: str = "auto",
    limit: int = 10,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Search MoNA online by compound name, InChIKey, SPLASH, or raw RSQL.

    If the remote filtered endpoint is not anonymously available, the response
    includes direct MoNA browser/download links plus the exact RSQL query that
    can be used after login or against a downloaded MoNA export.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Empty MoNA query."}
    limit = max(1, min(int(limit or 10), 100))
    rsql = build_mona_rsql(query, field=field)
    params = {"query": rsql, "size": limit, "page": 0}
    headers = {
        "Accept": "application/json",
        "User-Agent": "AnnoMASS/1.0",
    }
    auth_token = token or os.environ.get("MONA_API_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    search_url = f"{MONA_SEARCH_ENDPOINT}?{urlencode(params)}"
    browser_url = f"{MONA_BROWSER_URL}?query={quote(rsql, safe='')}"
    try:
        resp = requests.get(MONA_SEARCH_ENDPOINT, params=params, headers=headers, timeout=30)
        if resp.status_code in (401, 403):
            return _blocked_result(query, rsql, search_url, browser_url, resp.status_code)
        resp.raise_for_status()
        payload = resp.json()
        records = payload if isinstance(payload, list) else payload.get("content", [])
        summaries = [_summarize_record(record) for record in records[:limit]]
        return {
            "query": query,
            "field": field,
            "rsql": rsql,
            "engine": "mona_rest_search",
            "n_matches": len(summaries),
            "matches": summaries,
            "search_url": search_url,
            "browser_url": browser_url,
            "downloads_url": MONA_DOWNLOADS_URL,
            "note": "MoNA online filtered REST search returned records.",
        }
    except requests.RequestException as exc:
        return _fallback_result(query, rsql, search_url, browser_url, f"{type(exc).__name__}: {exc}")
    except ValueError as exc:
        return _fallback_result(query, rsql, search_url, browser_url, f"Invalid JSON from MoNA: {exc}")


def build_mona_rsql(query: str, field: str = "auto") -> str:
    field = (field or "auto").lower()
    if field == "rsql":
        return query
    if field == "splash" or query.lower().startswith("splash"):
        return f"splash:'{_quote_value(query)}'"
    if field == "inchikey" or re.fullmatch(r"[A-Z]{14}(?:-[A-Z]{10}-[A-Z])?", query):
        return f"exists(compound.metaData.name:'InChIKey' and compound.metaData.value~'{_quote_value(query)}')"
    if field == "precursor_mz":
        return f"exists(metaData.name:'precursor m/z' and metaData.value:'{_quote_value(query)}')"
    return f"exists(compound.names.name~'*{_quote_value(query)}*')"


def _quote_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _blocked_result(query: str, rsql: str, search_url: str, browser_url: str, status_code: int) -> Dict[str, Any]:
    return {
        "query": query,
        "rsql": rsql,
        "engine": "mona_rest_search",
        "n_matches": 0,
        "matches": [],
        "search_url": search_url,
        "browser_url": browser_url,
        "downloads_url": MONA_DOWNLOADS_URL,
        "status": "authorization_required",
        "note": (
            f"MoNA filtered REST search returned HTTP {status_code}. Open the browser URL "
            "or download a MoNA export and query it locally with query_massql_file."
        ),
    }


def _fallback_result(query: str, rsql: str, search_url: str, browser_url: str, reason: str) -> Dict[str, Any]:
    return {
        "query": query,
        "rsql": rsql,
        "engine": "mona_rest_search",
        "n_matches": 0,
        "matches": [],
        "search_url": search_url,
        "browser_url": browser_url,
        "downloads_url": MONA_DOWNLOADS_URL,
        "status": "fallback_links",
        "error": reason,
        "note": "MoNA API search was not available from this session; use the links or a local MoNA export.",
    }


def _summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = _meta_map(record.get("metaData") or [])
    compounds = record.get("compound") or []
    first_compound = compounds[0] if compounds else {}
    compound_meta = _meta_map(first_compound.get("metaData") or [])
    names = []
    for entry in first_compound.get("names") or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(entry.get("name"))
        elif isinstance(entry, str):
            names.append(entry)
    accession = record.get("id") or record.get("accession")
    return {
        "id": accession,
        "name": names[0] if names else None,
        "synonyms": names[1:5],
        "formula": first_compound.get("formula") or compound_meta.get("molecular formula"),
        "inchi_key": first_compound.get("inchiKey") or compound_meta.get("InChIKey"),
        "splash": record.get("splash"),
        "precursor_mz": meta.get("precursor m/z"),
        "precursor_type": meta.get("precursor type"),
        "ion_mode": meta.get("ionization mode"),
        "ms_level": meta.get("ms level"),
        "instrument": meta.get("instrument") or meta.get("instrument type"),
        "collision_energy": meta.get("collision energy"),
        "retention_time": meta.get("retention time"),
        "num_peaks": _peak_count(record),
        "score": (record.get("score") or {}).get("score") if isinstance(record.get("score"), dict) else record.get("score"),
        "url": f"{MONA_BASE_URL}/#/spectra/display/{accession}" if accession else MONA_BROWSER_URL,
    }


def _meta_map(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict) and item.get("name") and item.get("value") is not None:
            out[str(item["name"])] = item["value"]
    return out


def _peak_count(record: Dict[str, Any]) -> Optional[int]:
    spectrum = record.get("spectrum")
    if not isinstance(spectrum, str) or not spectrum.strip():
        return None
    return sum(1 for line in spectrum.splitlines() if line.strip())
