"""Live PubMed literature search for the Flavonoid Agent, via NCBI
E-utilities (esearch + esummary + efetch). Read-only, network-dependent,
never used to filter/rank Pipeline candidates -- purely for helping
interpret/explain a compound or spectrum result with supporting literature.

IMPORTANT for callers/prompting: an abstract's claims are about a DIFFERENT
sample/instrument/study than the user's current spectrum. The agent's
system prompt (core/agent_tools.py) is responsible for telling the model to
label anything drawn from these abstracts as literature-based context, kept
visibly separate from the spectrum's own hard evidence (evidence_level,
DeepMASS/DreaMS/Spectrum Predictor/AnnoSM scores) -- this module just fetches text,
it doesn't (and can't) enforce how the model uses it.
"""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT = 15
ABSTRACT_MAX_CHARS = 1500


def _fetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    """Best-effort: returns {pmid: abstract_text}, missing entries just
    aren't in the dict (e.g. no-abstract records, or the efetch call
    failing entirely) -- never raises, since abstracts are a nice-to-have
    on top of the title/author/journal esummary already provides."""
    if not pmids:
        return {}
    try:
        resp = requests.get(
            EFETCH_URL,
            params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
            timeout=TIMEOUT,
            headers={"User-Agent": "AnnoMASS-Agent/1.0"},
        )
        if resp.status_code != 200:
            return {}
        root = ET.fromstring(resp.content)
    except Exception:
        return {}

    abstracts = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        parts = []
        for ab_text in article.findall(".//Abstract/AbstractText"):
            text = "".join(ab_text.itertext()).strip()
            if not text:
                continue
            label = ab_text.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        if parts:
            full = " ".join(parts)
            if len(full) > ABSTRACT_MAX_CHARS:
                full = full[:ABSTRACT_MAX_CHARS] + "…"
            abstracts[pmid_el.text] = full
    return abstracts


def search_pubmed(query: str, max_results: int = 5) -> Dict[str, Any]:
    if not query or not query.strip():
        return {"error": "Empty query."}
    max_results = max(1, min(int(max_results), 20))

    try:
        search_resp = requests.get(
            ESEARCH_URL,
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results},
            timeout=TIMEOUT,
            headers={"User-Agent": "AnnoMASS-Agent/1.0"},
        )
    except requests.RequestException as exc:
        return {"error": f"PubMed search request failed: {exc}"}
    if search_resp.status_code != 200:
        return {"error": f"PubMed esearch returned HTTP {search_resp.status_code}"}
    try:
        pmids: List[str] = search_resp.json()["esearchresult"]["idlist"]
    except (ValueError, KeyError) as exc:
        return {"error": f"Unexpected PubMed esearch response shape: {exc}"}
    if not pmids:
        return {"query": query, "hits": []}

    try:
        summary_resp = requests.get(
            ESUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
            timeout=TIMEOUT,
            headers={"User-Agent": "AnnoMASS-Agent/1.0"},
        )
    except requests.RequestException as exc:
        return {"error": f"PubMed summary request failed: {exc}"}
    if summary_resp.status_code != 200:
        return {"error": f"PubMed esummary returned HTTP {summary_resp.status_code}"}
    try:
        result = summary_resp.json()["result"]
    except (ValueError, KeyError) as exc:
        return {"error": f"Unexpected PubMed esummary response shape: {exc}"}

    abstracts = _fetch_abstracts(pmids)

    hits = []
    for pmid in pmids:
        doc = result.get(pmid)
        if not doc:
            continue
        authors = [a.get("name", "") for a in doc.get("authors", [])][:5]
        hits.append({
            "pmid": pmid,
            "title": doc.get("title", ""),
            "authors": authors,
            "journal": doc.get("fulljournalname") or doc.get("source", ""),
            "pub_date": doc.get("pubdate", ""),
            "abstract": abstracts.get(pmid),  # None if unavailable, never fabricated
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"query": query, "hits": hits}
