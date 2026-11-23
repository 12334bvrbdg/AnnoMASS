# -*- coding: utf-8 -*-
"""
Spectrum Predictor (ms-pred) 适配器 -- 替代 MassFormer 作为候选结构谱图预测算法。

工作流程：
  1. 将候选 SMILES 批量写入 JSON 请求文件
  2. 调用 scripts/icefrag/run_iceberg.py（在 ~/ms-pred 自带的 .venv 中运行，
     子进程模式，与 GUI 的 deepmass2 环境完全隔离）
  3. 读取 JSON 响应，解析预测谱图 + 每个峰对应的碎片原子位掩码
  4. 与实测谱图计算余弦相似度 -- 评分方法与旧版 MassFormer 适配器完全一致
     （_clean_peaks / _cosine_similarity 逐字复用，未改动）

碎片结构查看：render_fragment_svg() 额外调用同一个子进程脚本的 render 子命令，
把 (smiles, frag_id) 变成一张高亮子结构的 RDKit SVG，供 GUI 点击谱峰时展示。

路径配置：
  MSPRED_DIR        ms-pred 代码根目录
  MSPRED_PYTHON     ms-pred 自带 .venv 的解释器
  GEN_CHECKPOINT / INTEN_CHECKPOINT   Spectrum Predictor 预训练权重（MassSpecGym，msg_all）
"""

import os
import sys
import json
import hashlib
import tempfile
import subprocess
import numpy as np

# ── 路径配置 ────────────────────────────────────────────────────
from core.paths import external_dir as _external_dir, env_python as _env_python, ENV_PREDICTION
MSPRED_DIR = str(_external_dir("ms-pred"))
_legacy_py = os.path.join(MSPRED_DIR, ".venv", "bin", "python")
MSPRED_PYTHON = os.environ.get("ANNOMASS_PREDICTION_PYTHON") or (
    _env_python(ENV_PREDICTION, None, "python") if os.path.exists(_env_python(ENV_PREDICTION, None, "python")) else _legacy_py)
BRIDGE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "icefrag", "run_iceberg.py",
)

GEN_CHECKPOINT = "models/msg_all/gen/best.ckpt"
INTEN_CHECKPOINT = "models/msg_all/inten_contr/best.ckpt"

# GPU is available on this machine (torch 2.11+cu128 in ms-pred's .venv).
# Verified 2026-07-27: predicting the same SMILES on CPU vs GPU gives
# cosine similarity 1.0 (identical peaks, identical intensities) across a
# 200-compound sample, and GPU cuts wall time ~3x on that sample (~0.39s
# to ~0.14s per structure) plus scales further at larger batch sizes
# (~0.045s/structure for a 1647-compound batch). Default to GPU; set
# DEEPMASS_ICEFRAG_GPU=0 to force CPU if the GPU is ever unavailable/busy.
USE_GPU = os.environ.get("DEEPMASS_ICEFRAG_GPU", "1").strip() == "1"

CACHE_DIR = os.environ.get(
    "ANNOMASS_ICEFRAG_CACHE_DIR",
    os.path.join(os.getcwd(), "data", "icefrag_cache"),
)

# Spectrum Predictor (msg_all / MassSpecGym) supports both polarities via
# ms_pred.common.chem_utils.ion2mass. Confirmed 1:1 name match with every
# adduct this project already uses, except [M+2H]2+, which Spectrum Predictor has no
# training data for -- same situation MassFormer was in, so it gets the
# same fallback treatment.
_POS_ADDUCTS = {"[M+H]+", "[M+H-H2O]+", "[M+Na]+", "[M+H-NH3]+", "[M+NH4]+"}
_NEG_ADDUCTS = {"[M-H]-"}
_UNSUPPORTED = {"[M+2H]2+"}


def _is_negative_adduct(adduct: str) -> bool:
    return str(adduct or "").strip().endswith("-")


def _effective_adduct(adduct: str) -> str:
    """Return the adduct Spectrum Predictor will actually be asked to predict."""
    if adduct in _NEG_ADDUCTS or adduct in _POS_ADDUCTS:
        return adduct
    if _is_negative_adduct(adduct):
        return adduct if adduct not in _UNSUPPORTED else "[M-H]-"
    return "[M+H]+"


def is_available() -> bool:
    """检查 Spectrum Predictor 是否可用（不加载模型）。"""
    try:
        gen_ckpt = os.path.join(MSPRED_DIR, GEN_CHECKPOINT)
        inten_ckpt = os.path.join(MSPRED_DIR, INTEN_CHECKPOINT)
        return (
            os.path.isfile(MSPRED_PYTHON)
            and os.path.isfile(BRIDGE_SCRIPT)
            and os.path.isfile(gen_ckpt)
            and os.path.isfile(inten_ckpt)
        )
    except Exception:
        return False


def neg_mode_available() -> bool:
    """Spectrum Predictor's msg_all checkpoint covers both polarities from one model."""
    return is_available()


# ── Cache ───────────────────────────────────────────────────────

def _cache_key(smiles: str, adduct: str, nce: float, instrument: str = "Orbitrap") -> str:
    raw = "|".join([
        "icefrag-v8-adduct-shift",
        _effective_adduct(adduct),
        f"{float(nce):.3f}",
        str(instrument or "Orbitrap"),
        str(smiles).strip(),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(smiles: str, adduct: str, nce: float, instrument: str = "Orbitrap") -> str:
    return os.path.join(CACHE_DIR, _cache_key(smiles, adduct, nce, instrument) + ".npz")


def _safe_int64_array(values):
    try:
        return np.asarray(values, np.int64)
    except (OverflowError, ValueError):
        # Fragment ids are atom-inclusion bitmasks. Very large glycosides can
        # exceed int64; scoring and benchmark metrics do not use these ids, so
        # keep the predicted spectrum and disable only fragment-click rendering.
        return np.array([], dtype=np.int64)


def _load_cached_prediction(smiles: str, adduct: str, nce: float, instrument: str = "Orbitrap"):
    path = _cache_path(smiles, adduct, nce, instrument)
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {
            "mz": np.asarray(data["mz"], float),
            "intensity": np.asarray(data["intensity"], float),
            "frag_ids": _safe_int64_array(data["frag_ids"]),
            "frag_bits": int(data["frag_bits"]),
            "canonical_smiles": str(data["canonical_smiles"]),
        }
    except Exception as e:
        print(f"[Spectrum Predictor] Cache read failed: {e}")
        return None


def _save_cached_prediction(smiles: str, adduct: str, nce: float, result: dict, instrument: str = "Orbitrap"):
    if len(result.get("mz", [])) == 0:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            _cache_path(smiles, adduct, nce, instrument),
            mz=np.asarray(result["mz"], float),
            intensity=np.asarray(result["intensity"], float),
            frag_ids=_safe_int64_array(result.get("frag_ids", [])),
            frag_bits=int(result.get("frag_bits", 0)),
            canonical_smiles=str(result.get("canonical_smiles", "")),
        )
    except Exception as e:
        print(f"[Spectrum Predictor] Cache write failed: {e}")


_EMPTY_RESULT = {
    "mz": np.array([]), "intensity": np.array([]),
    "frag_ids": np.array([], dtype=np.int64), "frag_bits": 0,
    "canonical_smiles": "",
}




# ── Peak cleaning / cosine similarity ──────────────────────────
# Copied verbatim from core/massformer_adapter.py so the scoring
# methodology is unchanged -- only the spectrum source (Spectrum Predictor instead of
# MassFormer) is new.

def _clean_peaks(mz, intensity, precursor_mz=0.0, top_k=80,
                 min_rel_intensity=0.01):
    """Normalize and keep informative MS/MS peaks for spectral scoring."""
    mz = np.asarray(mz, float)
    intensity = np.asarray(intensity, float)
    if len(mz) == 0 or len(intensity) == 0:
        return np.array([]), np.array([])

    n = min(len(mz), len(intensity))
    mz = mz[:n]
    intensity = intensity[:n]
    keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    if precursor_mz and precursor_mz > 0:
        keep &= (np.abs(mz - float(precursor_mz)) > 2.0)
    mz = mz[keep]
    intensity = intensity[keep]
    if len(mz) == 0:
        return np.array([]), np.array([])

    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    keep = intensity >= float(min_rel_intensity)
    mz = mz[keep]
    intensity = intensity[keep]
    if len(mz) == 0:
        return np.array([]), np.array([])

    if top_k and len(mz) > top_k:
        idx = np.argsort(intensity)[-int(top_k):]
        mz = mz[idx]
        intensity = intensity[idx]

    order = np.argsort(mz)
    return mz[order], np.sqrt(intensity[order])


def _cosine_similarity(mz_a, int_a, mz_b, int_b, tol_da=0.05,
                       precursor_mz=0.0, top_k=80,
                       min_rel_intensity=0.01):
    """Greedy peak-matched cosine for centroid MS/MS spectra."""
    mz_a  = np.asarray(mz_a,  float)
    int_a = np.asarray(int_a, float)
    mz_b  = np.asarray(mz_b,  float)
    int_b = np.asarray(int_b, float)

    if len(mz_a) == 0 or len(mz_b) == 0:
        return 0.0

    mz_a, int_a = _clean_peaks(
        mz_a, int_a, precursor_mz=precursor_mz,
        top_k=top_k, min_rel_intensity=min_rel_intensity,
    )
    mz_b, int_b = _clean_peaks(
        mz_b, int_b, precursor_mz=precursor_mz,
        top_k=top_k, min_rel_intensity=min_rel_intensity,
    )
    if len(mz_a) == 0 or len(mz_b) == 0:
        return 0.0

    pairs = []
    for i, m in enumerate(mz_a):
        lo = np.searchsorted(mz_b, m - tol_da, side="left")
        hi = np.searchsorted(mz_b, m + tol_da, side="right")
        for j in range(lo, hi):
            pairs.append((float(int_a[i] * int_b[j]), i, j))

    dot = 0.0
    used_a = set()
    used_b = set()
    for prod, i, j in sorted(pairs, reverse=True):
        if i in used_a or j in used_b:
            continue
        dot += prod
        used_a.add(i)
        used_b.add(j)

    na = float(np.sqrt(np.dot(int_a, int_a)))
    nb = float(np.sqrt(np.dot(int_b, int_b)))
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


def deepmet_entropy_cosine(
    mz_exp,
    int_exp,
    mz_pred,
    int_pred,
    precursor_mz,
    min_rel_intensity=0.01,
    top_k=80,
):
    """Reproduce DeepMet's two ICEBERG binned similarities.

    DeepMet processes the experimental spectrum with max-normalization and a
    square-root intensity transform, bins it by max at 0.1 Da, and compares it
    with ICEBERG's add-pooled prediction below precursor-1 Da.  Entropy is the
    normalized Jensen-Shannon similarity used by ``ms_pred.common.MassSpec``.
    """
    mz_exp = np.asarray(mz_exp, float)
    int_exp = np.asarray(int_exp, float)
    mz_pred = np.asarray(mz_pred, float)
    int_pred = np.asarray(int_pred, float)
    if not len(mz_exp) or not len(mz_pred):
        return 0.0, 0.0
    exp_keep = np.isfinite(mz_exp) & np.isfinite(int_exp) & (int_exp > 0)
    # process_spec_file first admits up to precursor+1, then MassSpec.similarity
    # applies ignore_mass=precursor-1 to both spectra.
    exp_keep &= mz_exp < float(precursor_mz) - 1.0
    pred_keep = np.isfinite(mz_pred) & np.isfinite(int_pred) & (int_pred > 0)
    pred_keep &= mz_pred < float(precursor_mz) - 1.0
    mz_exp, int_exp = mz_exp[exp_keep], int_exp[exp_keep]
    mz_pred, int_pred = mz_pred[pred_keep], int_pred[pred_keep]
    if not len(mz_exp) or not len(mz_pred):
        return 0.0, 0.0
    # Apply the same relative-intensity and strongest-peak filter to both
    # spectra. Raw instrument intensities are not comparable across files,
    # so the cutoff is relative to each spectrum's own base peak. Previously
    # this path admitted every positive-intensity peak, which systematically
    # depressed entropy similarity for dense/noisy experimental spectra.
    def _filter_peaks(mz, intensity):
        intensity = intensity / intensity.max()
        keep = intensity >= float(min_rel_intensity)
        mz, intensity = mz[keep], intensity[keep]
        if top_k and len(mz) > int(top_k):
            strongest = np.argsort(intensity)[-int(top_k):]
            mz, intensity = mz[strongest], intensity[strongest]
        return mz, intensity

    mz_exp, int_exp = _filter_peaks(mz_exp, int_exp)
    mz_pred, int_pred = _filter_peaks(mz_pred, int_pred)
    if not len(mz_exp) or not len(mz_pred):
        return 0.0, 0.0
    int_exp = np.sqrt(int_exp)
    num_bins, upper = 15000, 1500.0
    scale = (num_bins - 1) / upper

    def _bins(mz, intensity, pool):
        idx = np.floor(mz * scale).astype(np.int64) + 1
        valid = (idx >= 0) & (idx < num_bins)
        idx, intensity = idx[valid], intensity[valid]
        out = np.zeros(num_bins, dtype=np.float64)
        if pool == "add":
            np.add.at(out, idx, intensity)
        else:
            np.maximum.at(out, idx, intensity)
        return out

    exp = _bins(mz_exp, int_exp, "max")
    pred = _bins(mz_pred, int_pred, "add")
    cosine_denom = np.linalg.norm(exp) * np.linalg.norm(pred)
    cosine = float(np.dot(exp, pred) / cosine_denom) if cosine_denom > 0 else 0.0
    p, q = pred / pred.sum(), exp / exp.sum()
    mix = (p + q) / 2.0
    eps = 1e-22
    hp = -float(np.sum(p[p > 0] * np.log(p[p > 0] + eps)))
    hq = -float(np.sum(q[q > 0] * np.log(q[q > 0] + eps)))
    hm = -float(np.sum(mix[mix > 0] * np.log(mix[mix > 0] + eps)))
    entropy = 1.0 - (2.0 * hm - hp - hq) / np.log(4.0)
    return float(np.clip(entropy, 0, 1)), float(np.clip(cosine, 0, 1))


# ── Subprocess bridge ──────────────────────────────────────────

def _run_bridge(command: str, request: dict, timeout: float) -> dict:
    if not os.path.isfile(MSPRED_PYTHON) or not os.path.isfile(BRIDGE_SCRIPT):
        return {"error": "Spectrum Predictor bridge not available (ms-pred .venv or run_iceberg.py missing)"}
    with tempfile.TemporaryDirectory() as tmpdir:
        req_fp = os.path.join(tmpdir, "request.json")
        resp_fp = os.path.join(tmpdir, "response.json")
        with open(req_fp, "w") as f:
            json.dump(request, f)
        cmd = [
            MSPRED_PYTHON, BRIDGE_SCRIPT, command,
            "--input", req_fp, "--output", resp_fp,
            "--mspred-dir", MSPRED_DIR,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": f"Spectrum Predictor {command} timed out after {timeout}s"}
        except Exception as e:
            return {"error": f"Spectrum Predictor {command} subprocess error: {e}"}
        if not os.path.isfile(resp_fp):
            stderr_tail = (proc.stderr or "")[-800:]
            return {"error": f"Spectrum Predictor {command} produced no output.\n{stderr_tail}"}
        try:
            with open(resp_fp) as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to parse Spectrum Predictor {command} output: {e}"}


def predict_batch(smiles_list: list,
                  adduct: str = "[M+H]+",
                  nce: float = 40.0,
                  instrument: str = "Orbitrap",
                  timeout: int = 600) -> list:
    """批量预测多个 SMILES 的谱图（单次子进程调用，模型只加载一次）。

    instrument 必须是 ICEBERG 训练时用过的类别之一：Orbitrap / QTOF / IT-FT /
    Unknown（对应 ms_pred.common.chem_utils.instrument2onehot_pos）；传入其他
    值 predict_smis.py 会自动回退成 Orbitrap。

    返回：
        list of dict {"mz", "intensity", "frag_ids", "frag_bits",
        "canonical_smiles"}，长度与 smiles_list 相同；预测失败的条目返回空结果。
    """
    n = len(smiles_list)
    empty_list = [dict(_EMPTY_RESULT)] * n
    if n == 0 or not is_available():
        return empty_list

    eff_adduct = _effective_adduct(adduct)
    eff_instrument = str(instrument or "Orbitrap")
    results = [None] * n
    missing_positions = {}
    missing_ids = []
    for idx, smi in enumerate(smiles_list):
        cached = _load_cached_prediction(smi, eff_adduct, nce, eff_instrument)
        if cached is not None:
            results[idx] = cached
            continue
        if smi not in missing_positions:
            missing_positions[smi] = []
            missing_ids.append(smi)
        missing_positions[smi].append(idx)

    if not missing_ids:
        print(f"[Spectrum Predictor] Batch cache hit: {n}/{n}")
        return results

    cached_n = n - sum(len(v) for v in missing_positions.values())
    if cached_n:
        print(f"[Spectrum Predictor] Batch cache hit: {cached_n}/{n}; running {len(missing_ids)} new SMILES")

    request = {
        "candidates": [{"id": str(i), "smiles": smi} for i, smi in enumerate(missing_ids)],
        "adduct": eff_adduct,
        "nce": float(nce),
        "instrument": eff_instrument,
        "gpu": USE_GPU,
        "gen_checkpoint": GEN_CHECKPOINT,
        "inten_checkpoint": INTEN_CHECKPOINT,
    }
    response = _run_bridge("predict", request, timeout=timeout)
    if "error" in response:
        print(f"[Spectrum Predictor] Batch predict failed: {response['error']}")
        return [r if r is not None else dict(_EMPTY_RESULT) for r in results]

    by_id = response.get("results", {})
    for i, smi in enumerate(missing_ids):
        entry = by_id.get(str(i))
        if not entry or "error" in entry:
            pred = dict(_EMPTY_RESULT)
            if entry and "error" in entry:
                print(f"[Spectrum Predictor]   candidate failed: {entry['error']}")
        else:
            # Already merged by ms_pred.common.MassSpec.merged_spec() in
            # the bridge script -- see scripts/icefrag/run_iceberg.py.
            pred_mz = np.asarray(entry["mz"], float)
            pred_int = np.asarray(entry["intensity"], float)
            pred_frag = _safe_int64_array(entry.get("frag_ids", []))
            if len(pred_int) > 0 and pred_int.max() > 0:
                pred_int = pred_int / pred_int.max()
            pred = {
                "mz": pred_mz,
                "intensity": pred_int,
                "frag_ids": pred_frag,
                "frag_bits": int(entry["frag_bits"]),
                "canonical_smiles": entry["canonical_smiles"],
            }
            _save_cached_prediction(smi, eff_adduct, nce, pred, eff_instrument)
        for out_idx in missing_positions.get(smi, []):
            results[out_idx] = pred

    return [r if r is not None else dict(_EMPTY_RESULT) for r in results]


def predict_and_score(smiles: str,
                      query_mz, query_intensity,
                      precursor_mz: float,
                      adduct: str = "[M+H]+",
                      charge: int = 1,
                      nce: float = 40.0,
                      instrument: str = "Orbitrap",
                      tol_da: float = 0.02):
    """
    用 Spectrum Predictor 预测候选结构谱图并与查询谱图计算相似度。

    返回的 pred_mz/pred_int/frag_ids 是完整预测谱图（含前体/分子离子峰），
    供 GUI 显示用；余弦相似度内部单独用去掉前体峰（±2 Da）的副本计算，
    避免预测前体峰过强而虚高分数——这一处理只影响打分，不影响返回给调用方
    展示用的谱图数据。

    返回:
        score              (float)
        pred_mz            (np.ndarray)
        pred_int           (np.ndarray)
        frag_ids           (np.ndarray)  每个预测峰对应的碎片原子位掩码
        frag_bits          (int)         母体分子原子数（位掩码宽度）
        canonical_smiles   (str)         Spectrum Predictor 内部使用的规范化 SMILES
                                          （渲染碎片结构时必须原样复用）
    """
    if not is_available():
        return 0.0, np.array([]), np.array([]), np.array([]), 0, ""

    preds = predict_batch([smiles], adduct=adduct, nce=nce, instrument=instrument, timeout=120)
    result = preds[0] if preds else dict(_EMPTY_RESULT)

    pred_mz = np.asarray(result["mz"], float)
    pred_int = np.asarray(result["intensity"], float)
    frag_ids = np.asarray(result.get("frag_ids", []), np.int64)

    score_mz, score_int = pred_mz, pred_int
    if len(score_mz) > 0 and precursor_mz > 0:
        keep = np.abs(score_mz - precursor_mz) > 2.0
        score_mz = score_mz[keep]
        score_int = score_int[keep]

    score = _cosine_similarity(
        np.asarray(query_mz, float),
        np.asarray(query_intensity, float),
        score_mz, score_int,
        tol_da=tol_da,
        precursor_mz=precursor_mz,
    )

    return (
        score, pred_mz, pred_int,
        frag_ids,
        int(result.get("frag_bits", 0)),
        str(result.get("canonical_smiles", "")),
    )


def render_fragment_svg(smiles: str, frag_id: int, observed_mz: float = None,
                        size: int = 320, timeout: float = 30.0) -> dict:
    """
    渲染碎片高亮结构图（供 GUI 点击谱峰时调用）。

    ``smiles`` 必须是 predict_and_score / predict_batch 返回的
    ``canonical_smiles``（而不是原始候选 SMILES），否则原子编号可能与
    ``frag_id`` 位掩码不一致。

    返回:
        {"svg": str, "formula": str, "h_shift": int} 或 {"error": str}
    """
    request = {"smiles": smiles, "frag_id": int(frag_id), "size": int(size)}
    if observed_mz is not None:
        request["mz"] = float(observed_mz)
    response = _run_bridge("render", request, timeout=timeout)
    if "error" in response:
        print(f"[Spectrum Predictor] Fragment render failed: {response['error']}")
    return response
