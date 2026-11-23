"""Small local MassQL-style query engine for loaded AnnoMASS spectra.

This is intentionally a pragmatic subset of MassQL for interactive use in the
GUI/web/Agent. It supports the common diagnostic queries we need locally:
product ions (MS2PROD), neutral losses (MS2NL), precursor m/z (MS1MZ or
PRECURSORMZ), retention time (RT), and simple options such as TOLERANCEMZ,
TOLERANCEPPM, MININT, and LIMIT.
"""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MassQLCondition:
    kind: str
    value: float
    tol_mz: Optional[float] = None
    tol_ppm: Optional[float] = None
    min_intensity: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _get_mz_intensity(spectrum):
    mz = np.asarray(getattr(spectrum, "mz", []), dtype=float)
    inten = np.asarray(getattr(spectrum, "intensities", []), dtype=float)
    if len(mz) != len(inten):
        peaks = getattr(spectrum, "peaks", None)
        if peaks is not None:
            mz = np.asarray(getattr(peaks, "mz", []), dtype=float)
            inten = np.asarray(getattr(peaks, "intensities", []), dtype=float)
    return mz, inten


def _metadata(spectrum, key: str, default=None):
    try:
        return spectrum.get(key, default)
    except Exception:
        return getattr(spectrum, "metadata", {}).get(key, default)


def _rt_minutes(spectrum) -> Optional[float]:
    for key in ("retention_time", "rt", "scan_start_time"):
        val = _float_or_none(_metadata(spectrum, key))
        if val is not None:
            # Many MGF/matchms sources store retention_time in seconds; keep
            # already-small values as minutes, convert large values.
            return val / 60.0 if val > 100 else val
    return None


def _title(spectrum, index: int) -> str:
    return (
        _metadata(spectrum, "compound_name")
        or _metadata(spectrum, "title")
        or _metadata(spectrum, "scans")
        or f"Spectrum {index}"
    )


def _parse_options(option_text: str) -> Dict[str, float]:
    opts: Dict[str, float] = {}
    for name, raw in re.findall(r"([A-Za-z_]+)\s*=\s*([-+]?\d+(?:\.\d+)?)", option_text or ""):
        opts[name.upper()] = float(raw)
    return opts


def parse_query(query: str) -> Dict[str, Any]:
    """Parse a practical MassQL-like query string.

    Supported examples:
      QUERY scaninfo(MS2DATA) WHERE MS2PROD = 271.0601:TOLERANCEMZ=0.01
      WHERE MS2NL = 162.0528:TOLERANCEPPM=10
      MS1MZ = 303.05:TOLERANCEMZ=0.02
      RT BETWEEN 3.0 AND 5.5
    """
    text = (query or "").strip()
    if not text:
        raise ValueError("Empty MassQL query.")

    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", text, re.I)
    limit = int(limit_match.group(1)) if limit_match else 200

    where = text
    m = re.search(r"\bWHERE\b(.+)", text, re.I | re.S)
    if m:
        where = m.group(1)
    where = re.sub(r"\bLIMIT\s+\d+\b", "", where, flags=re.I)

    conditions: List[MassQLCondition] = []
    parts = re.split(r"\bAND\b", where, flags=re.I)
    i = 0
    while i < len(parts):
        part = parts[i].strip().strip(";")
        if not part:
            i += 1
            continue
        # Re-join "RT BETWEEN x AND y" split by the generic AND splitter.
        if re.search(r"\bBETWEEN\b", part, re.I) and i + 1 < len(parts):
            part = part + " AND " + parts[i + 1].strip()
            i += 1
        conditions.append(_parse_condition(part))
        i += 1
    if not conditions:
        raise ValueError("No supported MassQL condition found.")
    return {"conditions": conditions, "limit": limit}


def _parse_condition(part: str) -> MassQLCondition:
    between = re.match(
        r"^(RT|RETENTIONTIME|MS1MZ|PRECURSORMZ)\s+BETWEEN\s+([-+]?\d+(?:\.\d+)?)\s+AND\s+([-+]?\d+(?:\.\d+)?)",
        part,
        re.I,
    )
    if between:
        kind = between.group(1).upper()
        low, high = float(between.group(2)), float(between.group(3))
        return MassQLCondition(kind=kind, value=(low + high) / 2, low=min(low, high), high=max(low, high))

    m = re.match(
        r"^(MS2PROD|MS2PRODUCT|MS2NL|NEUTRALLOSS|MS1MZ|PRECURSORMZ|RT|RETENTIONTIME)\s*=\s*([-+]?\d+(?:\.\d+)?)(.*)$",
        part,
        re.I,
    )
    if not m:
        raise ValueError(f"Unsupported MassQL condition: {part}")
    kind = m.group(1).upper()
    opts = _parse_options(m.group(3))
    return MassQLCondition(
        kind=kind,
        value=float(m.group(2)),
        tol_mz=opts.get("TOLERANCEMZ") or opts.get("TOLMZ"),
        tol_ppm=opts.get("TOLERANCEPPM") or opts.get("TOLPPM"),
        min_intensity=opts.get("MININT") or opts.get("MININTENSITY"),
    )


def run_query(spectra: Iterable[Any], query: str, default_tol_mz: float = 0.01, limit: Optional[int] = None) -> Dict[str, Any]:
    parsed = parse_query(query)
    conditions: List[MassQLCondition] = parsed["conditions"]
    max_rows = int(limit or parsed.get("limit") or 200)
    rows: List[Dict[str, Any]] = []
    for idx, spectrum in enumerate(spectra):
        matched = [_match_condition(spectrum, cond) for cond in conditions]
        if all(m is not None for m in matched):
            row = _summary_row(idx, spectrum)
            for mi, detail in enumerate(matched, start=1):
                row.update({f"match_{mi}_{k}": v for k, v in detail.items()})
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return {
        "query": query,
        "conditions": [c.kind for c in conditions],
        "n_matches": len(rows),
        "matches": rows,
        "note": "Local MassQL subset: supports MS2PROD, MS2NL, MS1MZ/PRECURSORMZ, RT, tolerances, MININT, LIMIT.",
    }


def run_official_file_query(input_filename: str, query: str, limit: int = 200) -> Dict[str, Any]:
    """Run the installed official MassQL engine on a local MGF/mzML/mzXML file."""
    from massql import msql_engine

    p = Path(input_filename)
    if not p.exists():
        raise FileNotFoundError(f"MassQL input file not found: {input_filename}")
    df = msql_engine.process_query(
        query,
        str(p),
        cache="feather",
        cache_dir="/tmp/annomass_massql_cache",
    )
    return _df_result(query, str(p), df, limit=limit, engine="official_massql")


def run_online_query(source_url: str, query: str, limit: int = 200, max_mb: int = 300) -> Dict[str, Any]:
    """Download a remote MS data file and query it with the official MassQL engine.

    source_url should be a direct HTTPS/HTTP URL to a supported file, such as
    .mgf, .mzML, .mzXML, .json, .txt, or .dat. GNPS/MassIVE links work when
    they resolve to a downloadable data file rather than an HTML landing page.
    """
    local_path = download_remote_ms_file(source_url, max_mb=max_mb)
    result = run_official_file_query(local_path, query, limit=limit)
    result["source_url"] = source_url
    result["downloaded_file"] = local_path
    return result


def query_gnps_massql_service(query: str) -> Dict[str, Any]:
    """Use the public GNPS2 MassQL Sandbox service.

    This does not download data. It asks the GNPS2 MassQL web service to parse
    the query and generate the ProteoSAFe MSQL-NF workflow link that runs the
    query remotely on GNPS infrastructure.
    """
    import requests

    base = "https://massql.gnps2.org"
    session = requests.Session()
    parse_payload = {
        "output": "..output_parse.children..",
        "outputs": [{"id": "output_parse", "property": "children"}],
        "inputs": [{"id": "query", "property": "value", "value": query}],
        "changedPropIds": ["query.value"],
        "state": [],
    }
    link_payload = {
        "output": "..query_gnps_link.href..",
        "outputs": [{"id": "query_gnps_link", "property": "href"}],
        "inputs": [{"id": "query", "property": "value", "value": query}],
        "changedPropIds": ["query.value"],
        "state": [],
    }
    parse_resp = session.post(base + "/_dash-update-component", json=parse_payload, timeout=30)
    parse_resp.raise_for_status()
    link_resp = session.post(base + "/_dash-update-component", json=link_payload, timeout=30)
    link_resp.raise_for_status()
    parse_json = parse_resp.json()
    link_json = link_resp.json()
    children = (
        parse_json.get("response", {})
        .get("output_parse", {})
        .get("children", [])
    )
    translations = _extract_massql_translations(children)
    workflow_link = (
        link_json.get("response", {})
        .get("query_gnps_link", {})
        .get("href")
    )
    return {
        "query": query,
        "service": base,
        "workflow": "MSQL-NF",
        "workflow_link": workflow_link,
        "gnps_library_search": "https://massql-analysis.gnps2.org/MassQL_GNPS_Library_Searching",
        "massql_sandbox": base + "/",
        "translations": translations,
        "note": (
            "This uses the GNPS2 MassQL service online. It parses the query and "
            "returns the remote ProteoSAFe MSQL-NF workflow link; running a full "
            "GNPS job may require opening/submitting the workflow page."
        ),
    }


def query_online_library_router(query: str, libraries: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prepare online MassQL routes across GNPS/MassIVE-style resources.

    GNPS2 currently exposes the MassQL Sandbox/ProteoSAFe workflow link in a
    public, scriptable way. Interactive library-search/compendium apps are
    Streamlit/portal workflows, so this returns actionable links and the
    parsed query rather than pretending to have synchronous search results
    from every library.
    """
    selected = [s.lower() for s in (libraries or ["gnps", "massive", "libraries", "compendium", "mona"])]
    service = query_gnps_massql_service(query)
    q = quote(query, safe="")
    routes = []
    if "gnps" in selected or "massive" in selected:
        routes.append({
            "source": "GNPS/MassIVE ProteoSAFe MassQL workflow",
            "kind": "remote_workflow",
            "url": service.get("workflow_link"),
            "status": "ready_to_submit",
            "notes": "Runs MassQL remotely through the MSQL-NF workflow; task submission may require the GNPS page.",
        })
    if "libraries" in selected or "gnps" in selected:
        routes.append({
            "source": "GNPS2 MassQL Library Searching",
            "kind": "interactive_library_search",
            "url": "https://massql-analysis.gnps2.org/MassQL_GNPS_Library_Searching",
            "status": "interactive",
            "notes": "Use this for online spectral-library searching across GNPS-connected library resources.",
        })
    if "compendium" in selected:
        routes.append({
            "source": "MassQL Compendium",
            "kind": "query_examples",
            "url": "https://massql.gnps2.org/compendium/",
            "status": "interactive",
            "notes": "Useful for adapting community MassQL query templates for interpretation.",
        })
    if "mona" in selected:
        routes.append({
            "source": "MoNA",
            "kind": "external_spectral_library",
            "url": "https://mona.fiehnlab.ucdavis.edu/#/spectra/search",
            "downloads_url": "https://mona.fiehnlab.ucdavis.edu/#/downloads",
            "status": "external",
            "notes": (
                "MoNA is an online spectral library, not a MassQL-native endpoint in this app. "
                "Use query_mona_online for direct MoNA lookup, or download a MoNA export and run "
                "query_massql_file on the local library file."
            ),
        })
    routes.append({
        "source": "MassQL Sandbox",
        "kind": "parser_visualizer",
        "url": f"https://massql.gnps2.org/?query={q}",
        "status": "ready",
        "notes": "Online query parser/visualizer; confirms syntax and explains the query.",
    })
    return {
        "query": query,
        "requested_libraries": selected,
        "routes": routes,
        "translations": service.get("translations", {}),
        "workflow_link": service.get("workflow_link"),
        "note": (
            "MassQL can target multiple online spectral resources through GNPS2/ProteoSAFe and "
            "interactive library-search apps. Some resources require web workflow submission and "
            "do not expose synchronous result APIs."
        ),
    }


def _extract_massql_translations(children: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(children, list):
        return out
    current = None
    for child in children:
        if not isinstance(child, dict):
            continue
        props = child.get("props", {})
        ctype = child.get("type")
        text = props.get("children")
        if ctype == "H4" and isinstance(text, str):
            current = text.replace(" translation", "").strip().lower()
        elif ctype == "Pre" and current and isinstance(text, str):
            out[current] = text
            current = None
    return out


def download_remote_ms_file(source_url: str, max_mb: int = 300) -> str:
    import requests

    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("source_url must be http(s).")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".mgf", ".mzml", ".mzxml", ".json", ".txt", ".dat"}:
        # Some GNPS/MassIVE download endpoints do not put the filename in the
        # path. Keep going, but use .mgf as the safest generic suffix.
        suffix = ".mgf"
    out_dir = Path(tempfile.gettempdir()) / "annomass_massql_downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"remote_{int(time.time())}_{abs(hash(source_url)) % 10_000_000}{suffix}"
    max_bytes = int(max_mb) * 1024 * 1024
    got = 0
    with requests.get(source_url, stream=True, timeout=(10, 120), headers={"User-Agent": "AnnoMASS-MassQL/1.0"}) as resp:
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/html" in ctype:
            raise ValueError("URL returned HTML, not a downloadable MS data file. Use a direct MGF/mzML/mzXML download URL.")
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                got += len(chunk)
                if got > max_bytes:
                    raise ValueError(f"Remote file exceeds max_mb={max_mb}.")
                f.write(chunk)
    return str(out_path)


def _df_result(query: str, source: str, df: pd.DataFrame, limit: int = 200, engine: str = "massql") -> Dict[str, Any]:
    if df is None:
        df = pd.DataFrame()
    out = df.head(limit).copy()
    out = out.where(pd.notna(out), None)
    return {
        "query": query,
        "source": source,
        "engine": engine,
        "n_matches": int(len(df)),
        "matches": out.to_dict(orient="records"),
        "columns": list(df.columns),
        "note": "Official MassQL engine result. Use direct downloadable MS data URLs for online queries.",
    }


def _summary_row(index: int, spectrum) -> Dict[str, Any]:
    mz, inten = _get_mz_intensity(spectrum)
    return {
        "spectrum_index": index,
        "title": str(_title(spectrum, index)),
        "precursor_mz": _float_or_none(_metadata(spectrum, "precursor_mz")),
        "rt_min": _rt_minutes(spectrum),
        "ionmode": _metadata(spectrum, "ionmode") or _metadata(spectrum, "ion_mode"),
        "adduct": _metadata(spectrum, "adduct") or _metadata(spectrum, "precursor_type"),
        "n_peaks": int(len(mz)),
        "max_intensity": float(np.max(inten)) if len(inten) else None,
    }


def _tol(cond: MassQLCondition) -> float:
    if cond.tol_mz is not None:
        return cond.tol_mz
    if cond.tol_ppm is not None:
        return abs(cond.value) * cond.tol_ppm / 1_000_000.0
    return 0.01


def _match_condition(spectrum, cond: MassQLCondition) -> Optional[Dict[str, Any]]:
    kind = cond.kind
    if kind in {"MS2PROD", "MS2PRODUCT"}:
        return _match_peak(spectrum, cond, "product")
    if kind in {"MS2NL", "NEUTRALLOSS"}:
        return _match_peak(spectrum, cond, "neutral_loss")
    if kind in {"MS1MZ", "PRECURSORMZ"}:
        precursor = _float_or_none(_metadata(spectrum, "precursor_mz"))
        if precursor is None:
            return None
        if cond.low is not None and cond.high is not None:
            ok = cond.low <= precursor <= cond.high
        else:
            ok = abs(precursor - cond.value) <= _tol(cond)
        return {"type": kind, "target": cond.value, "observed": precursor, "error": precursor - cond.value} if ok else None
    if kind in {"RT", "RETENTIONTIME"}:
        rt = _rt_minutes(spectrum)
        if rt is None:
            return None
        if cond.low is not None and cond.high is not None:
            ok = cond.low <= rt <= cond.high
        else:
            ok = abs(rt - cond.value) <= _tol(cond)
        return {"type": kind, "target": cond.value, "observed": rt, "error": rt - cond.value} if ok else None
    return None


def _match_peak(spectrum, cond: MassQLCondition, mode: str) -> Optional[Dict[str, Any]]:
    mz, inten = _get_mz_intensity(spectrum)
    if len(mz) == 0:
        return None
    values = mz
    precursor = _float_or_none(_metadata(spectrum, "precursor_mz"))
    if mode == "neutral_loss":
        if precursor is None:
            return None
        values = precursor - mz
    diff = np.abs(values - cond.value)
    hit_idx = np.where(diff <= _tol(cond))[0]
    if cond.min_intensity is not None and len(hit_idx):
        hit_idx = np.array([i for i in hit_idx if inten[i] >= cond.min_intensity], dtype=int)
    if len(hit_idx) == 0:
        return None
    best = int(hit_idx[np.argmax(inten[hit_idx])])
    return {
        "type": "MS2NL" if mode == "neutral_loss" else "MS2PROD",
        "target": cond.value,
        "observed": float(values[best]),
        "peak_mz": float(mz[best]),
        "intensity": float(inten[best]),
        "error": float(values[best] - cond.value),
    }
