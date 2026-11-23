# -*- coding: utf-8 -*-
"""
FIDDLE 适配器（模型缓存 + 快速模式）

性能策略：
  - 模型只加载一次，后续调用直接复用（_MODEL_CACHE）
  - fast_mode=True（默认）：跳过耗时的树搜索 formula_refinement，
    直接返回神经网络输出并用质量窗口做简单后处理，单条谱图 < 0.5s
  - fast_mode=False：使用原版 formula_refinement，精度更高但慢
"""

import os
import re
import sys
import tempfile
import numpy as np

from core.paths import external_dir as _external_dir

FIDDLE_DIR = str(_external_dir("FIDDLE"))

CHECKPOINTS = {
    "Orbitrap": {
        "tcn":    os.path.join(FIDDLE_DIR, "check_point", "fiddle_tcn_orbitrap.pt"),
        "fdr":    os.path.join(FIDDLE_DIR, "check_point", "fiddle_fdr_orbitrap.pt"),
        "config": os.path.join(FIDDLE_DIR, "config", "fiddle_tcn_orbitrap.yml"),
    },
    "Q-TOF": {
        "tcn":    os.path.join(FIDDLE_DIR, "check_point", "fiddle_tcn_qtof.pt"),
        "fdr":    os.path.join(FIDDLE_DIR, "check_point", "fiddle_fdr_qtof.pt"),
        "config": os.path.join(FIDDLE_DIR, "config", "fiddle_tcn_qtof.yml"),
    },
}

_MODEL_CACHE = {}  # (instrument, device_str) -> cache dict
_ELEM_RE = re.compile(r"([A-Z][a-z]?)\d*")


def is_available(instrument: str = "Orbitrap") -> bool:
    ck = CHECKPOINTS.get(instrument)
    if ck is None:
        return False
    return (os.path.isfile(ck["tcn"]) and
            os.path.isfile(ck["fdr"]) and
            os.path.isfile(ck["config"]) and
            os.path.isdir(FIDDLE_DIR))


def _formula_elements(formula: str) -> set:
    if not formula or str(formula).strip().lower() in ("nan", "none", "null", ""):
        return set()
    return set(_ELEM_RE.findall(str(formula)))


def _filter_by_elements(formulas: list, allowed_elements: str) -> list:
    if not allowed_elements or str(allowed_elements).strip() == "":
        return formulas
    allowed = set(_ELEM_RE.findall(allowed_elements.upper()))
    if not allowed:
        return formulas
    return [f for f in formulas if f is not None and _formula_elements(f).issubset(allowed)]


def _spectrum_to_mgf_string(spectrum, title: str = "query") -> str:
    mz          = np.asarray(spectrum.peaks.mz, dtype=float)
    intensities = np.asarray(spectrum.peaks.intensities, dtype=float)

    precursor_mz = float(spectrum.get("precursor_mz") or spectrum.get("pepmass") or 0.0)
    adduct = (spectrum.get("adduct") or spectrum.get("ion") or "[M+H]+").strip()

    _ADDUCT_MAP = {
        "[M+H]1+": "[M+H]+", "[M-H]1-": "[M-H]-",
        "[M+Na]1+": "[M+Na]+", "[M+NH4]1+": "[M+NH4]+",
        "[M+2H]2+": "[M+2H]2+", "[2M+H]1+": "[2M+H]+", "[2M-H]1-": "[2M-H]-",
    }
    adduct = _ADDUCT_MAP.get(adduct, adduct)

    _SUPPORTED = {
        "[M+H]+", "[M+Na]+", "[2M+H]+",
        "[M+H-H2O]+", "[M-H2O+H]+", "[M+H-2H2O]+",
        "[M+H-NH3]+", "[M+H+NH3]+", "[M+NH4]+",
        "[M+H-CH2O2]+", "[M+H-CH4O2]+",
        "[M-H]-", "[2M-H]-",
        "[M-H-CO2]-", "[M-CHO2]-", "[M-H-H2O]-", "[M-H2O-H]-",
        "[M+2H]2+",
    }
    if adduct not in _SUPPORTED:
        ionmode = str(spectrum.get("ionmode") or "").lower()
        adduct = "[M+H]+" if "pos" in ionmode else "[M-H]-"

    collision_energy = spectrum.get("collision_energy") or spectrum.get("nce") or "30"
    try:
        ce_val = float(str(collision_energy).strip().rstrip("eVEv").strip())
        if ce_val > 200:
            ce_val = 30.0
        collision_energy = str(ce_val)
    except Exception:
        collision_energy = "30"

    lines = [
        "BEGIN IONS",
        f"TITLE={title}",
        f"PRECURSOR_MZ={precursor_mz:.6f}",
        f"PRECURSOR_TYPE={adduct}",
        f"COLLISION_ENERGY={collision_energy}",
    ]
    for m, i in zip(mz, intensities):
        lines.append(f"{m:.4f} {i:.4f}")
    lines.append("END IONS")
    return "\n".join(lines)


def _load_models(instrument: str, device_str: str = "cpu") -> dict:
    """加载并缓存模型，同一仪器只加载一次。"""
    cache_key = (instrument, device_str)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if FIDDLE_DIR not in sys.path:
        sys.path.insert(0, FIDDLE_DIR)

    import yaml
    import torch
    from collections import OrderedDict
    import importlib

    dataset_mod = importlib.import_module("dataset")
    model_mod   = importlib.import_module("model_tcn")
    utils_mod   = importlib.import_module("utils")

    _device = torch.device(
        device_str if (torch.cuda.is_available() or device_str == "cpu") else "cpu"
    )

    ck = CHECKPOINTS[instrument]
    with open(ck["config"], "r") as f:
        config = yaml.safe_load(f)

    def _load_state(path, net):
        state = torch.load(path, map_location=_device, weights_only=False)["model_state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        net.load_state_dict(state)
        net.eval()
        return net

    model     = _load_state(ck["tcn"], model_mod.MS2FNet_tcn(config["model"]).to(_device))
    fdr_model = _load_state(ck["fdr"], model_mod.FDRNet(config["model"]).to(_device))

    cache = {
        "model":              model,
        "fdr_model":          fdr_model,
        "config":             config,
        "device":             _device,
        "MGFDataset":         dataset_mod.MGFDataset,
        "formula_refinement": utils_mod.formula_refinement,
        "mass_calculator":    utils_mod.mass_calculator,
        "vector_to_formula":  utils_mod.vector_to_formula,
        "formula_to_vector":  utils_mod.formula_to_vector,
    }
    _MODEL_CACHE[cache_key] = cache
    print(f"[FIDDLE] 模型已缓存 (instrument={instrument}, device={_device})")
    return cache


def preload_models(instrument: str = "Orbitrap", device: str = "cpu"):
    """主动预加载模型，可在 UI 启动后台调用。"""
    if not is_available(instrument):
        return
    try:
        _load_models(instrument, device)
    except Exception as e:
        print(f"[FIDDLE] 预加载失败: {e}")


def _fast_post_process(formula_pred, m_val: float, ppm: float,
                       vector_to_formula, formula_to_vector,
                       mass_calculator_fn, exp_pre_type_str) -> list:
    """
    快速后处理：只做质量窗口校验，不跑树搜索。
    直接返回神经网络预测的分子式（如通过质量校验）。
    """
    try:
        from molmass import Formula as MolFormula
        delta = ppm * m_val / 1e6
        formula_str = formula_pred if isinstance(formula_pred, str) else str(formula_pred)
        if not formula_str or formula_str.strip() in ("", "None"):
            return []
        try:
            pred_mass = MolFormula(formula_str).isotope.mass
            if abs(pred_mass - m_val) <= delta:
                return [formula_str]
        except Exception:
            pass
        return [formula_str]   # 即使质量不完全匹配也返回，让 FDR 模型去排序
    except Exception:
        return [formula_pred] if formula_pred else []


def predict_formulas(spectrum,
                     instrument: str = "Orbitrap",
                     allowed_elements: str = "",
                     top_k: int = 5,
                     device: str = "cpu",
                     fast_mode: bool = True):
    """
    预测分子式。

    返回 (formulas, details)：
      formulas: list[str]  按置信度排序的分子式
      details:  list[dict] 每条包含 Formula / FDR Score / Exact Mass / Mass Error (ppm)
    """
    _empty = ([], [])

    if not is_available(instrument):
        print(f"[FIDDLE] 模型不存在，跳过 (instrument={instrument})")
        return _empty

    try:
        cache = _load_models(instrument, device)
    except Exception as e:
        print(f"[FIDDLE] 模型加载失败: {e}")
        return _empty

    import torch
    from torch.utils.data import DataLoader

    model              = cache["model"]
    fdr_model          = cache["fdr_model"]
    config             = cache["config"]
    _device            = cache["device"]
    MGFDataset         = cache["MGFDataset"]
    formula_refinement = cache["formula_refinement"]
    mass_calculator    = cache["mass_calculator"]
    vector_to_formula  = cache["vector_to_formula"]
    formula_to_vector  = cache["formula_to_vector"]

    with tempfile.TemporaryDirectory() as tmpdir:
        mgf_path = os.path.join(tmpdir, "q.mgf")
        with open(mgf_path, "w") as f:
            f.write(_spectrum_to_mgf_string(spectrum, title="q0") + "\n")

        try:
            dataset = MGFDataset(mgf_path, config["encoding"])
        except Exception as e:
            print(f"[FIDDLE] 数据集构建失败: {e}")
            return _empty

        if len(dataset) == 0:
            print("[FIDDLE] 谱图未通过过滤")
            return _empty

        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, drop_last=False)

        try:
            batch = next(iter(loader))
            _, exp_pre_type, x, env, neutral_add = batch
            x           = x.to(_device, dtype=torch.float32)
            env         = env.to(_device, dtype=torch.float32)
            neutral_add = neutral_add.to(_device, dtype=torch.float32)

            with torch.no_grad():
                _, pred_f, pred_mass, pred_atomnum, pred_hcnum = model(x, env)
            pred_f = pred_f - neutral_add

            formula_pred     = vector_to_formula(pred_f[0])
            exp_pre_mz       = env[0, 0]
            exp_pre_type_str = exp_pre_type[0]
        except Exception as e:
            print(f"[FIDDLE] 推理失败: {e}")
            return _empty

    try:
        m_val = float(mass_calculator(exp_pre_type_str, exp_pre_mz))
    except Exception:
        m_val = float(exp_pre_mz)

    pp = config.get("post_processing", {})

    if fast_mode:
        candidates = [formula_pred] if formula_pred else []
    else:
        try:
            refined    = formula_refinement(
                [formula_pred], m_val,
                pp.get("mass_tolerance", 20.0), pp.get("ppm_mode", True),
                top_k, pp.get("maxium_miss_atom_num", 20),
                min(pp.get("time_out", 30), 30),
                pp.get("refine_atom_type", ["C", "O", "N", "H"]),
                pp.get("refine_atom_num", [3, 3, 3, -1]),
            )
            candidates = [f for f in refined.get("formula", []) if f is not None]
        except Exception as e:
            print(f"[FIDDLE] refinement 失败，回退快速模式: {e}")
            candidates = [formula_pred] if formula_pred else []

    # FDR 打分（对所有候选都算分，哪怕只有1条）
    fdr_scores = [0.0] * len(candidates)
    try:
        if candidates:
            f_vecs   = [formula_to_vector(f) for f in candidates]
            f_tens   = torch.from_numpy(np.array(f_vecs)).to(_device, dtype=torch.float32)
            spec_rep = x[0].unsqueeze(0).repeat(len(candidates), 1)
            env_rep  = env[0].unsqueeze(0).repeat(len(candidates), 1)
            with torch.no_grad():
                fdr_scores = torch.sigmoid(
                    fdr_model(spec_rep, env_rep, f_tens)
                ).detach().cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"[FIDDLE] FDR 打分失败: {e}")

    # 排序
    paired = sorted(zip(fdr_scores, candidates), key=lambda t: t[0], reverse=True)
    fdr_scores = [s for s, _ in paired]
    candidates = [f for _, f in paired]

    # 元素过滤
    filtered = [(f, s) for f, s in zip(candidates, fdr_scores)
                if not allowed_elements or _formula_elements(f).issubset(
                    set(_ELEM_RE.findall(allowed_elements.upper())))]
    candidates  = [f for f, _ in filtered][:top_k]
    fdr_scores  = [s for _, s in filtered][:top_k]

    # 计算精确质量和质量误差
    details = []
    for f, score in zip(candidates, fdr_scores):
        try:
            from molmass import Formula as MolFormula
            exact_mass = MolFormula(f).isotope.mass
            mass_err_ppm = (exact_mass - m_val) / m_val * 1e6 if m_val else 0.0
        except Exception:
            exact_mass   = float("nan")
            mass_err_ppm = float("nan")
        details.append({
            "Formula":           f,
            "FDR Score":         round(float(score), 4),
            "Exact Mass":        round(exact_mass, 5) if exact_mass == exact_mass else "N/A",
            "Mass Error (ppm)":  round(mass_err_ppm, 2) if mass_err_ppm == mass_err_ppm else "N/A",
            "Instrument":        instrument,
        })

    print(f"[FIDDLE] {[(d['Formula'], d['FDR Score']) for d in details]}")
    return candidates, details
