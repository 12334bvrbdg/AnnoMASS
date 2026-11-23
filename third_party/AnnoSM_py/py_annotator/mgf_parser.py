"""
MGF (Mascot Generic Format) 谱图文件解析器
============================================
支持读取 AnnoSM / DeepMASS2 使用的标准 MGF 格式。

每个谱图块格式:
    BEGIN IONS
    PEPMASS=123.456
    CHARGE=1+
    TITLE=...
    RTINSECONDS=12.34
    100.0  5000.0
    200.0  300.0
    END IONS

返回: list[dict], 每个 dict 包含:
    - title: str          谱图标题 / ID
    - precursor_mz: float 前体离子 m/z
    - charge: int         电荷
    - rt_seconds: float   保留时间（秒）
    - peaks: ndarray      (N, 2) 的 m/z 和强度数组
"""

import re
import numpy as np
from pathlib import Path

# 正则：只匹配数字开头的行（峰数据行）
_PEAK_LINE_RE = re.compile(r"^\s*([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)")


def read_mgf(filepath, encoding="utf-8", errors="replace"):
    """
    读取 MGF 文件，返回谱图列表。

    Parameters
    ----------
    filepath : str or Path
        MGF 文件路径。
    encoding : str
        文件编码，默认 utf-8。
    errors : str
        编码错误处理方式，默认 'replace'。

    Returns
    -------
    list[dict]
        谱图字典列表，每个字典包含:
        - title: str
        - precursor_mz: float
        - charge: int (默认 1)
        - rt_seconds: float (默认 0.0)
        - peaks: np.ndarray of shape (N, 2), dtype float64
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MGF 文件不存在: {filepath}")

    spectra = []
    current = None
    in_ions = False

    with open(filepath, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            stripped = line.strip()

            # 跳过空行
            if not stripped:
                continue

            # BEGIN IONS → 开始新谱图
            if stripped.upper().startswith("BEGIN IONS"):
                current = {
                    "title": "",
                    "precursor_mz": None,
                    "charge": 1,
                    "rt_seconds": 0.0,
                    "peaks": [],
                    "meta": {},
                }
                in_ions = True
                continue

            # END IONS → 结束当前谱图
            if stripped.upper().startswith("END IONS"):
                if current is not None:
                    if current["precursor_mz"] is not None:
                        current["peaks"] = np.array(current["peaks"], dtype=np.float64)
                        spectra.append(current)
                    current = None
                    in_ions = False
                continue

            if not in_ions:
                continue

            # 解析 PEPMASS
            if stripped.upper().startswith("PEPMASS"):
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    try:
                        current["precursor_mz"] = float(parts[1].strip())
                    except ValueError:
                        pass
                continue

            # 解析 CHARGE
            if stripped.upper().startswith("CHARGE"):
                charge_str = stripped.split("=", 1)[1].strip()
                # 处理 "1+"、"1-"、"2+" 等格式
                match = re.match(r"([+-]?\d+)", charge_str)
                if match:
                    charge_val = int(match.group(1))
                    if "-" in charge_str or "negative" in charge_str.lower():
                        charge_val = -abs(charge_val)
                    else:
                        charge_val = abs(charge_val)
                    current["charge"] = charge_val
                continue

            # 解析 TITLE
            if stripped.upper().startswith("TITLE"):
                title = stripped.split("=", 1)[1].strip()
                current["title"] = title
                continue

            # 解析 RTINSECONDS
            if stripped.upper().startswith("RTINSECONDS"):
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    try:
                        current["rt_seconds"] = float(parts[1].strip())
                    except ValueError:
                        pass
                continue

            # 其它 KEY=VALUE 头部字段（INCHIKEY / SPECTRUMID / USI 等）存入 meta
            if "=" in stripped and not _PEAK_LINE_RE.match(stripped):
                key, val = stripped.split("=", 1)
                current["meta"][key.strip().upper()] = val.strip()
                continue

            # 峰数据行: "m/z    intensity"
            m = _PEAK_LINE_RE.match(stripped)
            if m:
                mz = float(m.group(1))
                intensity = float(m.group(2))
                if current is not None:
                    current["peaks"].append([mz, intensity])

    # 确保所有谱图都有正确的 peaks 类型
    for spec in spectra:
        if isinstance(spec["peaks"], list):
            spec["peaks"] = np.array(spec["peaks"], dtype=np.float64)

    return spectra


def read_mgf_single(filepath, spectrum_index=0, encoding="utf-8", errors="replace"):
    """
    读取 MGF 文件中指定索引的单个谱图（默认第一个）。

    Returns
    -------
    dict or None
        单个谱图，或 None（如果文件为空）。
    """
    spectra = read_mgf(filepath, encoding=encoding, errors=errors)
    if not spectra:
        return None
    if spectrum_index >= len(spectra):
        raise IndexError(
            f"谱图索引 {spectrum_index} 超出范围（共 {len(spectra)} 个谱图）"
        )
    return spectra[spectrum_index]


def spectrum_to_dataframe(spectra):
    """
    将谱图列表转换为 pandas DataFrame 格式。

    Parameters
    ----------
    spectra : list[dict]
        read_mgf 返回的谱图列表。

    Returns
    -------
    pd.DataFrame
        每行一个谱图，包含 title, precursor_mz, charge, rt_seconds, num_peaks, peaks 列。
    """
    import pandas as pd

    rows = []
    for spec in spectra:
        rows.append(
            {
                "title": spec["title"],
                "precursor_mz": spec["precursor_mz"],
                "charge": spec["charge"],
                "rt_seconds": spec["rt_seconds"],
                "num_peaks": len(spec["peaks"]),
                "peaks": spec["peaks"],
            }
        )
    return pd.DataFrame(rows)
