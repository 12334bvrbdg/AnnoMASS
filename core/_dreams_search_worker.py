#!/usr/bin/env python3
"""DreaMS embedding search worker — runs inside the 'dreams' conda env.

Implements the exact workflow from the DreaMS 谱图对比.ipynb notebook:
  1. Load reference library with MSData.load()
  2. Compute DreaMS embeddings for query spectra
  3. Compute cosine similarity between query and reference embeddings
  4. Extract top-k candidates per query
  5. Compute Modified Cosine Similarity for each hit
  6. Write results to CSV

Usage:
    python _dreams_search_worker.py '<json_config>'

Config keys:
    mgf_path, ref_hdf5, top_n, cho_only, out_csv, batch_size, dreams_root,
    mz_tol_ppm
"""

import hashlib
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["dreams_root"])

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
from dreams.definitions import DREAMS_EMBEDDING, PRECURSOR_MZ, RT, ADDUCT
from dreams.utils.spectra import PeakListModifiedCosine

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
PROTON = 1.00727646688
ADDUCT_MASS = {
    "[M+H]+": PROTON,
    "[M]+": 0.0,
    "[M-H]-": -PROTON,
    "[M+NA]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+H-H2O]+": PROTON - 18.010565,
    "[M+HCOO]-": 44.997654,
    "[M+CH3COO]-": 59.013851,
    "[M-2H]2-": -2 * PROTON,
}


def is_cho(formula: str) -> bool:
    elems = {e for e, _ in FORMULA_RE.findall(str(formula or ""))}
    return bool(elems) and elems <= {"C", "H", "O"}


def clean_adduct(v) -> str:
    s = str(v or "").strip().upper()
    return "" if s.lower() in ("", "nan", "none", "null") else s


def neutral_mass(mz, adduct):
    try:
        mz = float(mz)
    except Exception:
        return np.nan
    cleaned = clean_adduct(adduct)
    add = ADDUCT_MASS.get(cleaned)
    if add is None:
        return np.nan
    m = re.search(r"\](\d*)([+-])$", cleaned)
    z = int(m.group(1) or "1") if m else 1
    return mz * max(abs(z), 1) - add


def ppm_mask(values, target, ppm):
    try:
        target = float(target)
        ppm = float(ppm)
    except Exception:
        return np.zeros(len(values), dtype=bool)
    if target <= 0 or ppm <= 0:
        return np.zeros(len(values), dtype=bool)
    vals = np.asarray(values, dtype=float)
    return np.isfinite(vals) & (np.abs(vals - target) / target * 1e6 <= ppm)


def _lib_cache_path(ref_hdf5: str) -> str:
    """Disk cache for query-independent library preprocessing (CHO mask +
    precursor/neutral masses). Keyed by path+size+mtime so editing or
    rebuilding the .hdf5 invalidates it automatically."""
    st = os.stat(ref_hdf5)
    raw = f"{os.path.abspath(ref_hdf5)}|{st.st_size}|{int(st.st_mtime)}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(ref_hdf5)
    return os.path.join(os.path.dirname(os.path.abspath(ref_hdf5)),
                        f".{base}.libcache_{h}.npz")


def main():
    mgf_path  = cfg["mgf_path"]
    ref_hdf5  = cfg["ref_hdf5"]
    top_n     = int(cfg.get("top_n", 10))
    cho_only  = bool(cfg.get("cho_only", False))
    out_csv   = cfg["out_csv"]
    batch_size= int(cfg.get("batch_size", 32))
    mz_tol_ppm = float(cfg.get("mz_tol_ppm", 0.0) or 0.0)
    tol_per_query = None
    if cfg.get("mz_tol_ppm_file"):
        with open(cfg["mz_tol_ppm_file"]) as fh:
            tol_per_query = json.load(fh)

    # ── 1. Load reference library ────────────────────────────────────────────
    print("[DreaMS] Loading reference library...", flush=True)
    msdata_lib = MSData.load(ref_hdf5)
    embs_lib   = msdata_lib[DREAMS_EMBEDDING]
    n_lib      = embs_lib.shape[0]
    print(f"[DreaMS] Library: {n_lib:,} spectra, embedding dim {embs_lib.shape[1]}", flush=True)

    # CHO mask + precursor/neutral masses depend only on the library, not on
    # the query -- they cost ~40s to rebuild for 231k spectra, so cache them
    # next to the .hdf5 and reuse across runs.
    cache_fp = _lib_cache_path(ref_hdf5)
    cho_mask = None
    lib_prec_all = None
    lib_neutral_all = None
    if os.path.isfile(cache_fp):
        try:
            z = np.load(cache_fp, allow_pickle=False)
            if int(z["n_lib"]) == n_lib:
                cho_mask        = np.asarray(z["cho_mask"], dtype=bool)
                lib_prec_all    = np.asarray(z["lib_prec"], dtype=float)
                lib_neutral_all = np.asarray(z["lib_neutral"], dtype=float)
                print(f"[DreaMS] Library cache hit: {cache_fp}", flush=True)
        except Exception as e:
            print(f"[DreaMS] Library cache unreadable ({e}); rebuilding", flush=True)
            cho_mask = lib_prec_all = lib_neutral_all = None

    if cho_mask is None:
        print("[DreaMS] Building CHO-only mask...", flush=True)
        cho_mask = np.array([
            is_cho(msdata_lib.get_values("FORMULA", j) or "")
            for j in range(n_lib)
        ])
        lib_prec_all = np.zeros(n_lib, dtype=float)
        lib_neutral_all = np.full(n_lib, np.nan, dtype=float)
        print("[DreaMS] Building MS1/adduct cache...", flush=True)
        for j in range(n_lib):
            try:
                lp = float(msdata_lib.get_values(PRECURSOR_MZ, j) or 0.0)
            except Exception:
                lp = 0.0
            try:
                la = msdata_lib.get_values(ADDUCT, j) or ""
            except Exception:
                la = ""
            lib_prec_all[j] = lp
            lib_neutral_all[j] = neutral_mass(lp, la)
        try:
            np.savez_compressed(
                cache_fp, n_lib=n_lib, cho_mask=cho_mask,
                lib_prec=lib_prec_all, lib_neutral=lib_neutral_all,
            )
            print(f"[DreaMS] Saved library cache: {cache_fp}", flush=True)
        except Exception as e:
            print(f"[DreaMS] Could not save library cache ({e})", flush=True)

    if cho_only:
        lib_indices = np.where(cho_mask)[0]
        embs_lib_filtered = embs_lib[lib_indices]
        print(f"[DreaMS] CHO-only filter: {len(lib_indices):,} spectra kept", flush=True)
    else:
        lib_indices = np.arange(n_lib)
        embs_lib_filtered = embs_lib

    lib_prec_filtered = lib_prec_all[lib_indices]
    lib_neutral_filtered = lib_neutral_all[lib_indices]

    # ── 2. Compute query embeddings ─────────────────────────────────────────
    print("[DreaMS] Computing query embeddings...", flush=True)
    msdata = MSData.load(mgf_path)
    embs   = dreams_embeddings(msdata, batch_size=batch_size, progress_bar=False)
    n_query = embs.shape[0]
    print(f"[DreaMS] Query embeddings: {embs.shape}", flush=True)
    emb_xy = np.zeros((n_query, 2), dtype=float)
    emb_cluster = np.zeros(n_query, dtype=int)
    if n_query >= 2:
        print("[DreaMS] Projecting query embeddings...", flush=True)
        try:
            import umap
            reducer = umap.UMAP(
                metric="cosine",
                min_dist=0.4,
                n_neighbors=max(2, min(50, n_query - 1)),
                random_state=777,
            )
            emb_xy = reducer.fit_transform(embs)
        except Exception as e:
            print(f"[DreaMS] UMAP unavailable, using PCA fallback: {e}", flush=True)
            try:
                from sklearn.decomposition import PCA
                emb_xy = PCA(n_components=2, random_state=777).fit_transform(embs)
            except Exception as e2:
                print(f"[DreaMS] Projection failed: {e2}", flush=True)
        try:
            from sklearn.cluster import KMeans
            n_clusters = max(2, min(12, int(np.sqrt(n_query))))
            emb_cluster = KMeans(
                n_clusters=n_clusters,
                random_state=777,
                n_init=10,
            ).fit_predict(embs)
        except Exception as e:
            print(f"[DreaMS] Clustering failed: {e}", flush=True)

    # ── 3+4. Chunked cosine similarity + result rows ─────────────────────────
    # Similarities are computed one query-chunk at a time instead of one huge
    # (n_query x n_lib) matrix: for a 2000-query run that matrix alone would
    # be ~1 GB of float64 and a needless memory spike. Row-wise results are
    # identical.
    cos_sim = PeakListModifiedCosine()
    rows = []
    QUERY_CHUNK = 256
    print(f"[DreaMS] Building result table for {n_query} queries...", flush=True)

    for chunk_start in range(0, n_query, QUERY_CHUNK):
        chunk_end = min(chunk_start + QUERY_CHUNK, n_query)
        sims = cosine_similarity(embs[chunk_start:chunk_end], embs_lib_filtered)

        for i_rel in tqdm(range(chunk_end - chunk_start), desc="Building results",
                          file=sys.stdout, initial=chunk_start, total=n_query):
            i = chunk_start + i_rel
            try:
                prec_mz = float(msdata.get_values(PRECURSOR_MZ, i) or 0.0)
            except Exception:
                prec_mz = 0.0
            try:
                rt = float(msdata.get_values(RT, i) or 0.0)
            except Exception:
                rt = 0.0
            try:
                q_adduct = str(msdata.get_values(ADDUCT, i) or "")
            except Exception:
                q_adduct = ""

            q_neutral = neutral_mass(prec_mz, q_adduct)
            tol = mz_tol_ppm
            if tol > 0 and tol_per_query is not None and i < len(tol_per_query):
                tol = float(tol_per_query[i])
            if tol > 0 and np.isfinite(q_neutral):
                ms1_mask = ppm_mask(lib_neutral_filtered, q_neutral, tol)
            elif tol > 0:
                ms1_mask = ppm_mask(lib_prec_filtered, prec_mz, tol)
            else:
                ms1_mask = np.ones(len(lib_indices), dtype=bool)

            candidate_local = np.where(ms1_mask)[0]
            if len(candidate_local) == 0:
                candidate_local = np.arange(len(lib_indices))
            n_take = min(top_n, len(candidate_local))
            topk_local = candidate_local[np.argsort(sims[i_rel, candidate_local])[-n_take:][::-1]]

            for rank0, local_j in enumerate(topk_local):
                j = int(lib_indices[int(local_j)])   # index in full library
                dreams_sim = float(sims[i_rel, int(local_j)])

                # Library metadata
                try:
                    lib_smiles = msdata_lib.get_smiles(j) or ""
                except Exception:
                    lib_smiles = ""
                try:
                    lib_id = str(msdata_lib.get_values("IDENTIFIER", j) or "")
                except Exception:
                    lib_id = ""
                try:
                    lib_prec_mz = float(msdata_lib.get_values(PRECURSOR_MZ, j) or 0.0)
                except Exception:
                    lib_prec_mz = 0.0
                try:
                    lib_adduct = str(msdata_lib.get_values(ADDUCT, j) or "")
                except Exception:
                    lib_adduct = ""
                try:
                    lib_ce = str(msdata_lib.get_values("COLLISION_ENERGY", j) or "")
                except Exception:
                    lib_ce = ""
                try:
                    lib_name = str(msdata_lib.get_values("name", j) or "")
                except Exception:
                    lib_name = ""
                try:
                    lib_formula = str(msdata_lib.get_values("FORMULA", j) or "")
                except Exception:
                    lib_formula = ""
                try:
                    lib_ikey = str(msdata_lib.get_values("INCHIKEY", j) or "")
                except Exception:
                    lib_ikey = ""

                # Library peaks (top 60, "mz:relative_intensity;...") so the GUI
                # can draw the matched library spectrum under the query.
                lib_peaks = ""
                try:
                    _ls = np.asarray(msdata_lib.get_spectra(j), dtype=float)
                    if _ls.ndim == 2 and _ls.shape[0] == 2 and _ls.shape[1] > 0:
                        _mz, _it = _ls[0], _ls[1]
                    elif _ls.ndim == 2 and _ls.shape[1] == 2:
                        _mz, _it = _ls[:, 0], _ls[:, 1]
                    else:
                        _mz, _it = np.array([]), np.array([])
                    if len(_it) and _it.max() > 0:
                        _keep = np.argsort(_it)[-60:]
                        _keep = _keep[np.argsort(_mz[_keep])]
                        lib_peaks = ";".join(
                            f"{_mz[k]:.4f}:{_it[k] / _it.max():.4f}" for k in _keep)
                except Exception:
                    lib_peaks = ""

                # Modified cosine similarity
                mod_cos = 0.0
                try:
                    spec_q   = msdata.get_spectra(i)
                    spec_lib = msdata_lib.get_spectra(j)
                    prec_q   = msdata.get_prec_mzs(i)
                    prec_l   = msdata_lib.get_prec_mzs(j)
                    mod_cos  = float(cos_sim(
                        spec1=spec_q, prec_mz1=prec_q,
                        spec2=spec_lib, prec_mz2=prec_l,
                    ))
                except Exception:
                    pass

                rows.append({
                    "feature_id":               i + 1,
                    "query_index":              i,
                    "precursor_mz":             prec_mz,
                    "RT":                       rt,
                    "topk":                     rank0 + 1,
                    "library_j":                j,
                    "library_SMILES":           lib_smiles,
                    "library_ID":               lib_id,
                    "library_name":             lib_name,
                    "library_formula":          lib_formula,
                    "library_inchikey":         lib_ikey,
                    "library_precursor_mz":     lib_prec_mz,
                    "library_adduct":           lib_adduct,
                    "library_collision_energy": lib_ce,
                    "library_peaks":            lib_peaks,
                    "DreaMS_similarity":        dreams_sim,
                    "Modified_cosine_similarity": mod_cos,
                    "ms1_filter_candidates":    int(len(candidate_local)),
                    "dreams_umap_x":            float(emb_xy[i, 0]),
                    "dreams_umap_y":            float(emb_xy[i, 1]),
                    "dreams_cluster":           int(emb_cluster[i]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[DreaMS] Wrote {len(df)} rows → {out_csv}", flush=True)


if __name__ == "__main__":
    main()
