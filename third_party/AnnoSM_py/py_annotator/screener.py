"""
TPIL (Precursor Ion Library) 前体离子筛选模块
==============================================

对应 AnnoSM.R 中的 preprocess.spec.com.fla() 函数逻辑。

步骤:
  1. 读取 TPIL（前体离子库），包含 Formula/OHsum/OCH3sum 等信息
  2. 对于输入 MGF 中的每个谱图:
     a) 前体离子 m/z 按 ppm 匹配 TPIL → 初步候选列表
     b) 如果有多个候选且 MGF 包含碎片离子:
        - 用碎片离子反向匹配 TPIL 中预计算的碎片质量
        - 进一步用匹配到的碎片对应的 OHsum/OCH3sum 组合精确筛选
  3. 输出筛选后的候选化合物列表
"""

import pandas as pd
import numpy as np
from pathlib import Path


class CompoundScreener:
    """
    基于 TPIL 的化合物筛选器。

    Parameters
    ----------
    tpil_path : str or Path
        TPIL Excel 文件路径。
    sheet_name : str
        Excel sheet 名称，默认第一个 sheet。
    """

    # TPIL 中预计算碎片的前 n 行（对应 R 代码中的 precursorion1[1:66, ]）
    # 这是 AnnoSM 中用于碎片反向匹配的参考表头范围
    _FRAG_REF_ROWS = 66

    def __init__(self, tpil_path, sheet_name=0):
        tpil_path = Path(tpil_path)
        if not tpil_path.exists():
            raise FileNotFoundError(f"TPIL 文件不存在: {tpil_path}")

        ext = tpil_path.suffix.lower()
        if ext in (".xls", ".xlsx"):
            self.tpil = pd.read_excel(tpil_path, sheet_name=sheet_name)
        elif ext == ".csv":
            self.tpil = pd.read_csv(tpil_path)
        else:
            raise ValueError(f"不支持的 TPIL 文件格式: {ext}，请使用 .xls/.xlsx/.csv")

        # 标准化列名
        self.tpil = self._normalize_columns(self.tpil)

        # 确保关键列存在
        required_cols = [
            "cal.m.z.H",
            "OHsum",
            "OCH3sum",
            "Formulas",
            "Compound group",
            "Class",
        ]
        for col in required_cols:
            if col not in self.tpil.columns:
                raise KeyError(f"TPIL 缺少关键列: '{col}'")

        # 四舍五入 cal.m.z.H 到 4 位小数（与 R 代码一致）
        self.tpil["cal.m.z.H"] = self.tpil["cal.m.z.H"].round(4)

    def _normalize_columns(self, df):
        """标准化列名：去除首尾空格，统一命名"""
        rename_map = {}
        for col in df.columns:
            new_col = col.strip()
            # 处理 "Sugar and phenylpropionyl units " 这种带尾部空格的列名
            if "Sugar and phenylpropionyl" in new_col or "Sugar" in new_col:
                new_col = "Sugar and phenylpropionyl units"
            rename_map[col] = new_col
        df = df.rename(columns=rename_map)
        return df

    def calc_ppm(self, exp_mz, theo_mz):
        """计算 ppm 误差"""
        return abs(exp_mz - theo_mz) / theo_mz * 1_000_000

    def screen_single(
        self,
        precursor_mz,
        fragment_peaks=None,
        precursor_ppm=10,
        frag_ppm=10,
        title="X1",
        intensity=None,
        rt=None,
        use_fragment_refinement=True,
    ):
        """
        对单个谱图进行前体离子筛选。

        Parameters
        ----------
        precursor_mz : float
            实验前体离子 m/z。
        fragment_peaks : np.ndarray or None
            (N, 2) 的碎片峰 [m/z, intensity] 数组。用于二级筛选。
        precursor_ppm : float
            前体离子 ppm 容差，默认 10。
        frag_ppm : float
            碎片离子 ppm 容差，默认 10。
        title : str
            谱图标题，默认 "X1"。
        intensity : float or None
            前体离子强度。
        rt : float or None
            保留时间。

        Returns
        -------
        pd.DataFrame
            筛选后的候选列表，包含 "Compound ID" 等列。
            如果没有匹配，返回空 DataFrame。
        """
        # --- Step 1: 前体离子直接匹配 ---
        self.tpil["_ppm"] = self.calc_ppm(precursor_mz, self.tpil["cal.m.z.H"])
        candidates = self.tpil[self.tpil["_ppm"] <= precursor_ppm].copy()

        if len(candidates) == 0:
            return pd.DataFrame()

        # 基础记录
        rec = {
            "Compound ID": title,
            "Exp.m/z": round(precursor_mz, 4),
            "Intensity": round(intensity, 4) if intensity is not None else None,
            "RT": round(rt, 4) if rt is not None else None,
        }

        # --- Step 2: 如果只有一个候选，直接返回 ---
        if len(candidates) == 1:
            result = candidates.copy()
            for k, v in rec.items():
                result[k] = v
            return result

        # --- Step 3: 多个候选，用碎片离子做二级筛选 ---
        if (
            use_fragment_refinement
            and fragment_peaks is not None
            and len(fragment_peaks) > 0
        ):
            # 3a: 用碎片匹配 TPIL 前 66 行的参考碎片
            frag_ref = self.tpil.head(self._FRAG_REF_ROWS).copy()

            matched_by_frag = pd.DataFrame()
            # The paper implementation checks every experimental fragment
            # against the first 66 TPIL aglycone rows.
            for frag_mz, _ in fragment_peaks:
                frag_ref["_frag_ppm"] = self.calc_ppm(frag_mz, frag_ref["cal.m.z.H"])
                frag_hits = frag_ref[frag_ref["_frag_ppm"] <= frag_ppm]
                if len(frag_hits) > 0:
                    matched_by_frag = pd.concat(
                        [matched_by_frag, frag_hits], ignore_index=True
                    )

            if len(matched_by_frag) > 0:
                # 3b: 从匹配到的碎片中提取 OHsum/OCH3sum 组合
                matched_oh_och3 = (
                    matched_by_frag[["OHsum", "OCH3sum"]]
                    .drop_duplicates()
                )

                # 3c: 用这些 OH/OCH3 组合过滤候选列表
                filtered = pd.DataFrame()
                for _, row in matched_oh_och3.iterrows():
                    oh = row["OHsum"]
                    och3 = row["OCH3sum"]
                    subset = candidates[
                        (candidates["OHsum"] == oh) & (candidates["OCH3sum"] == och3)
                    ]
                    filtered = pd.concat([filtered, subset], ignore_index=True)

                if len(filtered) > 0:
                    augmented = candidates[
                        candidates.get("_candidate_augmented", False).astype(bool)
                    ] if "_candidate_augmented" in candidates else candidates.iloc[0:0]
                    candidates = pd.concat([filtered, augmented], ignore_index=True).drop_duplicates()
                # 如果过滤后为空，回退到原始候选列表

        # 组装结果
        result = candidates.copy()
        for k, v in rec.items():
            result[k] = v
        return result

    def screen_mgf(
        self,
        spectra,
        precursor_ppm=10,
        frag_ppm=10,
        progress_callback=None,
        use_fragment_refinement=True,
    ):
        """
        对 MGF 文件中的所有谱图进行筛选。

        Parameters
        ----------
        spectra : list[dict]
            read_mgf() 返回的谱图列表。
        precursor_ppm : float
            前体离子 ppm 容差。
        frag_ppm : float
            碎片离子 ppm 容差。
        progress_callback : callable or None
            进度回调函数 f(current, total)。

        Returns
        -------
        pd.DataFrame
            所有筛选结果的合并 DataFrame。
        """
        all_results = []
        total = len(spectra)

        for idx, spec in enumerate(spectra):
            if progress_callback:
                progress_callback(idx + 1, total)

            title = spec.get("title", f"X{idx + 1}")
            if not title:
                title = f"X{idx + 1}"

            result = self.screen_single(
                precursor_mz=spec["precursor_mz"],
                fragment_peaks=spec["peaks"],
                precursor_ppm=precursor_ppm,
                frag_ppm=frag_ppm,
                title=title,
                intensity=(
                    spec["peaks"][0, 1]
                    if len(spec["peaks"]) > 0
                    else None
                ),
                rt=spec.get("rt_seconds", None),
                use_fragment_refinement=use_fragment_refinement,
            )
            if len(result) > 0:
                all_results.append(result)

        if len(all_results) == 0:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)
        # 去除 _ppm 辅助列
        if "_ppm" in combined.columns:
            combined = combined.drop(columns=["_ppm"])
        if "_frag_ppm" in combined.columns:
            combined = combined.drop(columns=["_frag_ppm"])
        return combined
