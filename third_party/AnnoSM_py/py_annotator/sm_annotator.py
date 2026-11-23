"""
SM (Substituent Mode) 取代基模式注释核心模块
===============================================

对应 AnnoSM.R 中的以下函数逻辑:
  - sm.spec.com.one.fla()       — 单个化合物的 SM 注释
  - sm.spec.com.all.fla()       — 全部化合物的 SM 注释
  - Classification of SMs       — FLA / ISO 分类
  - Annotation of OH Positions  — A 环 OH 位点注释 (A10 / A20)

核心算法流程:
  1. 读取筛选结果 (precursorion_filter) 和 CFIL (特征碎片离子库)
  2. 对每个实验碎片离子，在 CFIL 中按 ±50 ppm 匹配
  3. 按 "Substitution modes" 分组，聚合碎片丰度和 (sum_abund)
  4. 按 sum_abund 降序排列
  5. FLA/ISO 分类: 根据参考离子列表 (Q2/Q4/Q5/Q6/Q7/Q8) 判断
  6. A 环 OH 位点注释: A10 → A10(5/7) 或 A10(6/8); A20 → A20(5&7) 或 A20(6/8&#)
  7. 计算 Score = sum_abund / 总碎片丰度
"""

import pandas as pd
import numpy as np
from pathlib import Path


class SMAnnotator:
    """
    取代基模式 (Substituent Mode) 注释器。

    使用 CFIL (Characteristic Fragment Ion Library) 对已筛选的化合物
    进行取代基模式匹配和注释。

    Parameters
    ----------
    cfil_path : str or Path
        CFIL Excel 文件路径。
    tpil_path : str or Path, optional
        TPIL Excel 文件路径。如果提供，可跳过外部筛选步骤直接运行完整流程。
    cfil_sheet : str or int
        CFIL 的 sheet 名称或索引，默认 0。
    """

    # ==================== 参考离子列表（与 R 代码完全一致） ====================
    # Q2: B 环丢失系列（[M+H-B ring]+-CO 等）
    Q2_REF = [
        "[M+H-B ring]+-CO",
        "[M+H-B ring]+-2×CO",
        "[M+H-B ring]+-3×CO",
        "[M+H-B ring]+-4×CO",
        "[M+H-B ring]+-CH3-CO",
        "[M+H-B ring]+-CH3-2×CO",
        "[M+H-B ring]+-CH3-3×CO",
    ]
    # Q4: 0，2B+ 系列 → FLA 标志
    Q4_REF = ["0，2B+", "0，2B+-CO"]
    # Q5: 0，4B+ 系列
    Q5_REF = ["0，4B+-CO", "0，4B+-2×CO", "0，4B+-3×CO"]
    # Q6: 1，4B+ 系列
    Q6_REF = ["1，4B+", "1，4B+-CO", "1，4B+--2×CO"]
    # Q7: 0，3B+ 系列
    Q7_REF = ["0，3B+", "0，3B+-CO", "0，3B+-2×CO"]
    # Q8: [B ring]+ 系列
    Q8_REF = [
        "[B ring]+",
        "[B ring]+-CO",
        "[B ring]+-CH3",
        "[B ring]+-CH3-CO",
        "[B ring]+-CH3-2×CO",
    ]

    # 所有参考离子系列，用于分类
    Q_REF_GROUPS = {
        "Q2": Q2_REF,
        "Q4": Q4_REF,
        "Q5": Q5_REF,
        "Q6": Q6_REF,
        "Q7": Q7_REF,
        "Q8": Q8_REF,
    }

    def __init__(self, cfil_path, tpil_path=None, cfil_sheet=0):
        # 加载 CFIL
        cfil_path = Path(cfil_path)
        if not cfil_path.exists():
            raise FileNotFoundError(f"CFIL 文件不存在: {cfil_path}")

        ext = cfil_path.suffix.lower()
        if ext in (".xls", ".xlsx"):
            self.cfil = pd.read_excel(cfil_path, sheet_name=cfil_sheet)
        elif ext == ".csv":
            self.cfil = pd.read_csv(cfil_path)
        else:
            raise ValueError(f"不支持的 CFIL 文件格式: {ext}")

        # 标准化列名
        self.cfil = self._normalize_columns(self.cfil)

        # 验证必需列
        required = ["Substitution modes", "Cal.m/z of Frag", "OHsum", "OCH3sum", "Annotations"]
        for col in required:
            if col not in self.cfil.columns:
                raise KeyError(f"CFIL 缺少关键列: '{col}'. 现有列: {list(self.cfil.columns)}")

        # 预计算 CFIL 的 digit（四舍五入到 4 位小数）
        self.cfil["_digit"] = self.cfil["Cal.m/z of Frag"].round(4)
        mode_skeleton = {
            mode: self._sm_to_skeleton(mode)
            for mode in self.cfil["Substitution modes"].dropna().unique()
        }
        self.cfil["_mode_skeleton"] = self.cfil["Substitution modes"].map(
            mode_skeleton
        )
        self._specificity_cache = {}

        # 预构建 m/z 排序索引，用于 match_single_compound 的快速区间查询
        # 避免每次对 107 万行全表扫描算 ppm
        import bisect
        self._cfil_sorted = self.cfil.sort_values("_digit").reset_index(drop=True)
        self._cfil_mz_list = self._cfil_sorted["_digit"].tolist()
        self._cfil_bisect = bisect

        # 可选: 加载 TPIL（如果提供了路径）
        self.tpil = None
        if tpil_path is not None:
            from .screener import CompoundScreener
            self.screener = CompoundScreener(tpil_path)

    def _normalize_columns(self, df):
        """标准化列名，处理可能的空格和 Unicode 差异"""
        rename_map = {}
        for col in df.columns:
            new_col = col.strip()
            rename_map[col] = new_col
        return df.rename(columns=rename_map)

    def calc_ppm(self, exp_mz, theo_mz):
        """计算 ppm 误差"""
        return abs(exp_mz - theo_mz) / theo_mz * 1_000_000

    # ==================== 骨架一致性过滤 ====================

    # SM 前缀 -> 骨架类型 的映射表
    # 用于将 CFIL 中的 Substitution modes 名称映射回骨架类型
    # 注意: 空前缀的骨架 (Flavones/Flavonols) 需要通过 C 环编码区分
    _PREFIX_TO_SKELETON = {
        "Iso-": "IsoFlavones",
        "Fla": "Flavanone",
        "Fno": "Flavanonol",
        "F3o": "Flavan3ol",
        "Cha": "Chalcone",
        "Dhc": "Dihydrochalcone",
        "Ifn": "Isoflavanone",
        "Ifl": "Isoflavan",
        "Aur": "Aurone",
        "Ant": "Anthocyanidin",
        "Fva": "Flavan",
        "Pte": "Pterocarpan",
        "Xan": "Xanthone",
    }

    # 空前缀骨架的 C 环编码 -> 骨架类型 映射
    _CRING_TO_SKELETON = {
        "C00": "Flavones",
        "C10": "Flavonols (C-3OH)",
        "C01": "Flavonols (C-3OCH3)",
    }

    def _sm_to_skeleton(self, sm_name):
        """
        从 Substitution modes 名称推断骨架类型。

        SM 命名: <prefix>A<OH><OCH3>B<OH><OCH3>C<OH><OCH3>
        - 有前缀的 (Iso-/Fla/Fno/...) 直接映射
        - 无前缀的通过 C 环编码区分: C00=Flavones, C10=Flavonols(C-3OH), C01=Flavonols(C-3OCH3)

        Returns
        -------
        str or None
            骨架类型名称，无法判断返回 None。
        """
        if not sm_name or pd.isna(sm_name):
            return None

        sm_name = str(sm_name).strip()

        # 检查是否有已知前缀 (按长度降序匹配，避免 "Iso-" 被 "I" 误匹配)
        for prefix in sorted(self._PREFIX_TO_SKELETON.keys(), key=len, reverse=True):
            if sm_name.startswith(prefix):
                return self._PREFIX_TO_SKELETON[prefix]

        # 无前缀: 通过 C 环编码区分
        # 提取 C 环部分 (SM 末尾的 C?? )
        # SM 格式: ...C<OH><OCH3>，如 C00, C10, C01
        if len(sm_name) >= 3:
            c_code = sm_name[-3:]  # 如 "C00", "C10"
            if c_code in self._CRING_TO_SKELETON:
                return self._CRING_TO_SKELETON[c_code]

        return None

    def _skeleton_matches(self, sm_name, target_skeleton):
        """
        判断 SM 名称对应的骨架是否与目标骨架一致。

        对于 target_skeleton 为 None 的情况（原版 TPIL 无骨架信息），
        返回 True（不做过滤）。

        对于 Flavones/Flavonols 这类共享空前缀的骨架，
        需要精确匹配 C 环编码。
        """
        if target_skeleton is None or pd.isna(target_skeleton):
            return True

        sm_skeleton = self._sm_to_skeleton(sm_name)
        if sm_skeleton is None:
            return True  # 无法判断骨架时不过滤

        return sm_skeleton == target_skeleton

    def _get_specificity_weights(self, oh_sum, och3_sum, target_skeleton):
        """Return IDF-like fragment weights within a comparable SM group."""
        key = (int(oh_sum), int(och3_sum), target_skeleton)
        if key in self._specificity_cache:
            return self._specificity_cache[key]

        subset = self.cfil[
            (self.cfil["OHsum"] == key[0])
            & (self.cfil["OCH3sum"] == key[1])
        ]
        if target_skeleton is not None and not pd.isna(target_skeleton):
            subset = subset[subset["_mode_skeleton"] == target_skeleton]

        mode_count = subset["Substitution modes"].nunique()
        if mode_count <= 1:
            weights = pd.Series(1.0, index=subset["_digit"].unique())
        else:
            fragment_mode_counts = subset.groupby("_digit")[
                "Substitution modes"
            ].nunique()
            weights = np.log(mode_count / fragment_mode_counts) / np.log(mode_count)
        result = weights.to_dict()
        self._specificity_cache[key] = result
        return result

    def _count_unique_matched_fragments(self, fragment_peaks, precursor_info, fragment_ppm):
        """
        统计给定骨架/前体信息下，实验碎片与 CFIL 匹配的唯一实验 m/z 数量。

        用于骨架候选排序: 匹配碎片数越多，说明该骨架越能解释实验谱图。
        相比 sum_abund，碎片数不易受单个高强度巧合匹配的影响。

        Parameters
        ----------
        fragment_peaks : np.ndarray
            (N, 2) 的碎片峰 [m/z, intensity]。
        precursor_info : dict or Series
            包含 OHsum, OCH3sum, Skeleton。
        fragment_ppm : float
            碎片 ppm 容差。

        Returns
        -------
        int
            匹配到的唯一实验碎片数。
        """
        if len(fragment_peaks) == 0:
            return 0

        oh_sum = int(precursor_info.get("OHsum", 0))
        och3_sum = int(precursor_info.get("OCH3sum", 0))
        target_skeleton = precursor_info.get("Skeleton", None)

        # 筛选 CFIL 到当前 OH/OCH3/骨架
        cfil_subset = self.cfil[
            (self.cfil["OHsum"] == oh_sum) & (self.cfil["OCH3sum"] == och3_sum)
        ].copy()

        if target_skeleton is not None and not pd.isna(target_skeleton):
            cfil_subset = cfil_subset[
                cfil_subset["Substitution modes"].apply(
                    lambda sm: self._skeleton_matches(sm, target_skeleton)
                )
            ].copy()

        if len(cfil_subset) == 0:
            return 0

        matched_mzs = set()
        for mz_val, _ in fragment_peaks:
            digi = round(mz_val, 4)
            cfil_subset["_ppm"] = self.calc_ppm(digi, cfil_subset["_digit"])
            hits = cfil_subset[cfil_subset["_ppm"] <= fragment_ppm]
            if len(hits) > 0:
                matched_mzs.add(digi)

        return len(matched_mzs)

    def _build_diagnostic_fragment_table(self, oh_sum, och3_sum):
        """
        构建诊断碎片表: 骨架 -> {只在该骨架 CFIL 中出现的 m/z 集合}。

        诊断碎片 = 对于给定 (OHsum, OCH3sum) 组合，某个 m/z 值
        只在一种骨架的 CFIL 条目中出现，不在其他骨架中出现。
        匹配到诊断碎片意味着该骨架很可能正确。

        Parameters
        ----------
        oh_sum : int
        och3_sum : int

        Returns
        -------
        dict[str, set[float]]
            骨架名称 -> 诊断 m/z 值集合 (4位小数)
        """
        # 筛选 CFIL 到当前 OH/OCH3
        subset = self.cfil[
            (self.cfil["OHsum"] == oh_sum) & (self.cfil["OCH3sum"] == och3_sum)
        ]

        # 构建 m/z -> 出现过的骨架集合
        mz_to_skeletons = {}
        for _, row in subset.iterrows():
            mz = round(row["_digit"], 4)
            skel = self._sm_to_skeleton(row["Substitution modes"])
            if skel is None:
                continue
            if mz not in mz_to_skeletons:
                mz_to_skeletons[mz] = set()
            mz_to_skeletons[mz].add(skel)

        # 找出诊断碎片 (只属于一种骨架的 m/z)
        diagnostic = {}
        for mz, skeletons in mz_to_skeletons.items():
            if len(skeletons) == 1:
                skel = list(skeletons)[0]
                if skel not in diagnostic:
                    diagnostic[skel] = set()
                diagnostic[skel].add(mz)

        return diagnostic

    def _compute_diagnostic_score(self, fragment_peaks, diag_mzs, fragment_ppm):
        """
        计算诊断碎片置信度: 实验谱图中匹配到诊断碎片的强度总和。

        Parameters
        ----------
        fragment_peaks : np.ndarray
            (N, 2) 的碎片峰 [m/z, intensity]。
        diag_mzs : set[float]
            该骨架的诊断 m/z 值集合。
        fragment_ppm : float
            碎片 ppm 容差。

        Returns
        -------
        float
            诊断碎片匹配的总强度。
        """
        if not diag_mzs or len(fragment_peaks) == 0:
            return 0.0

        score = 0.0
        for mz_val, intensity in fragment_peaks:
            digi = round(mz_val, 4)
            for diag_mz in diag_mzs:
                ppm = self.calc_ppm(digi, diag_mz)
                if ppm <= fragment_ppm:
                    score += intensity
                    break  # 同一实验碎片只计一次

        return score

    # ==================== 单个化合物的 SM 匹配（核心） ====================

    def match_single_compound(
        self,
        compound_id,
        precursor_mz,
        fragment_peaks,
        precursor_info,
        fragment_ppm=50,
    ):
        """
        对单个化合物/谱图进行 SM 匹配。

        Parameters
        ----------
        compound_id : str
            化合物 ID (如 "X1")。
        precursor_mz : float
            前体离子 m/z（用于计算总丰度时的标识）。
        fragment_peaks : np.ndarray
            (N, 2) 的碎片峰 [m/z, intensity]。
        precursor_info : pd.Series or dict
            该化合物的前体信息，至少包含 OHsum, OCH3sum。
            如果有多组筛选结果，应对每组分别调用。
        fragment_ppm : float
            碎片离子 ppm 容差，默认 50。

        Returns
        -------
        pd.DataFrame
            SM 匹配结果，列:
            - Compound ID
            - Substitution modes
            - sum_abund
            - Score
            - Class (FLA/ISO)
            - OH/OCH3 等信息
        """
        abund_total = fragment_peaks[:, 1].sum() if len(fragment_peaks) > 0 else 0
        if abund_total == 0:
            return pd.DataFrame()

        # --- Step 1: 遍历所有碎片离子，在 CFIL 中匹配 ---
        # 用 bisect 在预排序的 m/z 索引上做区间查询，避免全表扫描
        combined_matches = []

        for mz_val, intensity in fragment_peaks:
            # 四舍五入到 4 位小数
            digi = round(mz_val, 4)
            # 计算 ppm 容差对应的 m/z 窗口
            tol = digi * fragment_ppm / 1e6
            lo = digi - tol
            hi = digi + tol
            # bisect 定位区间 [lo, hi] 在排序 m/z 列表中的范围
            left = self._cfil_bisect.bisect_left(self._cfil_mz_list, lo)
            right = self._cfil_bisect.bisect_right(self._cfil_mz_list, hi)
            if left >= right:
                continue
            hits = self._cfil_sorted.iloc[left:right].copy()
            # 精确过滤 (bisect 区间可能略宽)
            hits["_ppm"] = self.calc_ppm(digi, hits["_digit"])
            hits = hits[hits["_ppm"] <= fragment_ppm]

            if len(hits) == 0:
                continue

            # 给每条匹配记录加上当前碎片的丰度
            hits["_matched_abund"] = intensity
            hits["_matched_mz"] = digi
            combined_matches.append(hits)

        if len(combined_matches) == 0:
            return pd.DataFrame()

        combined = pd.concat(combined_matches, ignore_index=True)

        # --- Step 1b: 去重 - 每个 (实验m/z, Substitution modes) 只计一次丰度 ---
        # CFIL 中同一 SM 可能有多个不同 Annotation 的条目具有相同 m/z
        # (如 1,3A+ 和 0,2A+-CO 都是 m/z 137.0233)。
        # 原版 AnnoSM 的 CFIL 每个 SM 不重复 m/z，但扩展版存在重复。
        # 为避免同一实验碎片被多次计入 sum_abund，需按 (实验m/z, SM) 去重。
        combined = combined.drop_duplicates(
            subset=["_matched_mz", "Substitution modes"], keep="first"
        ).copy()

        # 获取当前化合物的 OHsum / OCH3sum
        oh_sum = int(precursor_info.get("OHsum", 0))
        och3_sum = int(precursor_info.get("OCH3sum", 0))

        # --- Step 2: 按 OHsum/OCH3sum 过滤 CFIL 匹配结果 ---
        # R 代码中的关键步骤:
        #   combined_CFIL2 <- merge(..., by = c("OHsum","OCH3sum"))
        #   combined_CFIL2OH <- subset(combined_CFIL2, OHsum == T1)
        #   combined_CFIL2F <- subset(combined_CFIL2OH, OCH3sum == T2)
        combined = combined[
            (combined["OHsum"] == oh_sum) & (combined["OCH3sum"] == och3_sum)
        ].copy()

        if len(combined) == 0:
            return pd.DataFrame()

        # --- Step 2b: 骨架一致性过滤 (扩展版新增) ---
        # 原版 AnnoSM 只覆盖 4 类黄酮，不需要骨架过滤。
        # 扩展版覆盖 16 类，不同骨架可能产生相同 m/z 的 RDA 碎片，
        # 导致错误骨架的 SM 因巧合匹配而排名过高。
        # 用 TPIL 筛选确定的骨架类型过滤 CFIL 的 SM 候选。
        target_skeleton = precursor_info.get("Skeleton", None)
        if target_skeleton is not None and not pd.isna(target_skeleton):
            combined = combined[
                combined["Substitution modes"].apply(
                    lambda sm: self._skeleton_matches(sm, target_skeleton)
                )
            ].copy()

        if len(combined) == 0:
            return pd.DataFrame()

        specificity_weights = self._get_specificity_weights(
            oh_sum, och3_sum, target_skeleton
        )
        combined["_specificity_weight"] = (
            combined["_digit"].map(specificity_weights).fillna(0.0)
        )
        combined["_specificity_abund"] = (
            combined["_matched_abund"] * combined["_specificity_weight"]
        )

        # --- Step 3: 按 Substitution modes 聚合 ---
        # 计算每个 SM 的总丰度
        sm_abund = (
            combined.groupby("Substitution modes")[[
                "_matched_abund", "_specificity_abund"
            ]]
            .sum()
            .reset_index()
        )
        sm_abund.columns = [
            "Substitution modes", "sum_abund", "specificity_abund"
        ]

        # --- Step 4: 按 sum_abund 降序排列 ---
        sm_abund = sm_abund.sort_values("sum_abund", ascending=False).reset_index(
            drop=True
        )

        # --- Step 5: FLA / ISO 分类 ---
        sm_abund["Class"] = self._classify_sm(combined, sm_abund, oh_sum, och3_sum)

        # --- Step 6: A 环 OH 位点注释 ---
        sm_abund = self._annotate_oh_positions(combined, sm_abund, oh_sum, och3_sum)

        # --- Step 7: 计算 Score ---
        sm_abund["Score"] = (sm_abund["sum_abund"] / abund_total).round(4)
        sm_abund["SpecificityScore"] = (
            sm_abund["specificity_abund"] / abund_total
        ).round(4)

        # 添加化合物信息
        sm_abund["Compound ID"] = compound_id
        sm_abund["OHsum"] = oh_sum
        sm_abund["OCH3sum"] = och3_sum

        # 列顺序
        cols = [
            "Compound ID",
            "Substitution modes",
            "sum_abund",
            "Score",
            "SpecificityScore",
            "Class",
            "OHsum",
            "OCH3sum",
        ]
        return sm_abund[cols]

    # ==================== FLA / ISO 分类 ====================

    def _classify_sm(self, combined_matches, sm_abund, oh_sum, och3_sum):
        """
        对每个 SM 进行 FLA (Flavone) 或 ISO (Isoflavone) 分类。

        对应 R 代码中的 "Classification of SMs based on CFIs" 部分。

        规则:
          - 如果 Q4 参考离子丰度和 > 0 → FLA
          - 如果 Q4 == 0 且 Q2/Q5/Q6/Q7/Q8 中 ≥4 组丰度和 > 0 → ISO
        """
        # 获取含 Annotations 的完整匹配数据
        merged = combined_matches.merge(
            sm_abund[["Substitution modes"]], on="Substitution modes", how="inner"
        )

        classes = {}
        for sm_name in sm_abund["Substitution modes"].unique():
            subset = merged[merged["Substitution modes"] == sm_name]

            # 解析 Annotations（用 "/" 分隔）
            all_annotations = []
            for annos in subset["Annotations"].dropna():
                parts = str(annos).split("/")
                all_annotations.extend([p.strip() for p in parts])

            # 统计每组参考离子的丰度和
            group_sums = {}
            for group_name, ref_list in self.Q_REF_GROUPS.items():
                group_subset = subset[
                    subset["Annotations"].apply(
                        lambda x: (
                            len(set(str(x).split("/")) & set(ref_list)) > 0
                            if pd.notna(x)
                            else False
                        )
                    )
                ]
                group_sums[group_name] = group_subset["_matched_abund"].sum()

            t4 = group_sums.get("Q4", 0)
            t2 = group_sums.get("Q2", 0)
            t5 = group_sums.get("Q5", 0)
            t6 = group_sums.get("Q6", 0)
            t7 = group_sums.get("Q7", 0)
            t8 = group_sums.get("Q8", 0)

            # 统计非零的组数 (Q2, Q5, Q6, Q7, Q8 — 即排除 Q4)
            non_zero_groups = sum(1 for v in [t2, t5, t6, t7, t8] if v > 0)

            if t4 > 0:
                classes[sm_name] = "FLA"
            elif non_zero_groups >= 4:
                classes[sm_name] = "ISO"
            else:
                classes[sm_name] = ""

        return sm_abund["Substitution modes"].map(classes)

    # ==================== A 环 OH 位点注释 ====================

    def _annotate_oh_positions(self, combined_matches, sm_abund, oh_sum, och3_sum):
        """
        对 SM 名称中的 A10 和 A20 模式进行 A 环 OH 位点注释。

        对应 R 代码中的 "Annotation of OH Positions on Ring A" 部分。

        规则:
          A10:
            - 同时存在 '1，3A+-CO' 和 '1，4A+' → A 类（用于判断 A10(5/7)）
            - 存在 '1，3A+' → B 类（用于判断 A10(6/8)）
            - 只有 A 类 → A10(5/7)  ;  只有 B 类 → A10(6/8)
            - 两者都有 → abund_A/abund_B < 0.3 → A10(5/7) 否则 A10(6/8)

          A20:
            - 类似逻辑，但 A 类 → A20(5&7), B 类 → A20(6/8&#)
        """
        merged = combined_matches.merge(
            sm_abund[["Substitution modes"]], on="Substitution modes", how="inner"
        )

        def _annotate_a_ring(sm_name, pattern, label_a, label_b):
            """对包含 pattern 的 SM 名进行 A 环注释"""
            subset = merged[merged["Substitution modes"] == sm_name]

            # 区分 A 类和 B 类匹配
            a_rows = []
            b_rows = []
            for _, row in subset.iterrows():
                annos = str(row.get("Annotations", ""))
                parts = set(annos.split("/"))

                has_13a_co = "1，3A+-CO" in parts
                has_14a = "1，4A+" in parts
                has_13a = "1，3A+" in parts

                if has_13a_co and has_14a:
                    a_rows.append(row)
                elif has_13a:
                    b_rows.append(row)

            abund_a = sum(r["_matched_abund"] for r in a_rows) if a_rows else 0
            abund_b = sum(r["_matched_abund"] for r in b_rows) if b_rows else 0

            if abund_a == 0:
                return sm_name.replace(pattern, label_a)
            if abund_b == 0:
                return sm_name.replace(pattern, label_b)
            # 两者都有
            if abund_a / abund_b < 0.3:
                return sm_name.replace(pattern, label_a)
            else:
                return sm_name.replace(pattern, label_b)

        updated_modes = []
        for sm_name in sm_abund["Substitution modes"]:
            updated = sm_name
            if "A10" in updated:
                updated = _annotate_a_ring(
                    updated, "A10", "A10(5/7)", "A10(6/8)"
                )
            if "A20" in updated:
                updated = _annotate_a_ring(
                    updated, "A20", "A20(5&7)", "A20(6/8&#)"
                )
            updated_modes.append(updated)

        sm_abund["Substitution modes"] = updated_modes
        return sm_abund

    # ==================== 骨架选择策略 ====================

    # 诊断碎片: 注释模式 -> 该碎片是哪些骨架的专属特征
    # 仅匹配骨架专属碎片，避免巧合匹配干扰
    _DIAGNOSTIC_EXCLUSIVE = {
        # Flavonol 专属: 3-OH 使 0,2A+ 包含 C3
        "0，2A+": {"Flavonols (C-3OH)"},
        "0,2A+": {"Flavonols (C-3OH)"},
        "1，2B+": {"Flavonols (C-3OH)"},
        "1,2B+": {"Flavonols (C-3OH)"},
        # Isoflavone 专属: B 环在 C3
        "0，3B+": {"IsoFlavones"},
        "0,3B+": {"IsoFlavones"},
        "1，4B+": {"IsoFlavones"},
        "1,4B+": {"IsoFlavones"},
        # Pterocarpan 专属: 稠环系统
        "Benzofuran+": {"Pterocarpan"},
        "Chroman+": {"Pterocarpan"},
        # Flavone 专属
        "Benzofuranone+": {"Flavones"},
        "Benzoyl+": {"Flavones"},
        "Xanthylium+": {"Flavones"},
        # Flavan3ol 专属
        "RDA": {"Flavan3ol"},
        # Chalcone 专属
        "B-ring ethyl+": {"Chalcone"},
        "A-ring acylium+": {"Chalcone"},
    }

    def _compute_idf_score(self, cfil_subset, fragment_peaks, frag_ppm,
                           exp_mz_counts):
        """
        计算 IDF 加权得分: 每个实验碎片被越少骨架匹配，权重越高。

        Parameters
        ----------
        cfil_subset : DataFrame
            当前骨架的 CFIL 子集。
        fragment_peaks : np.ndarray
            实验碎片峰。
        frag_ppm : float
            碎片 ppm 容差。
        exp_mz_counts : dict
            {round(exp_mz, 4): 骨架匹配数} 预计算。

        Returns
        -------
        float
            IDF 加权得分。
        """
        if len(fragment_peaks) == 0:
            return 0.0
        theo_mzs = cfil_subset["Cal.m/z of Frag"].values
        idf = 0.0
        for exp_mz, exp_inten in fragment_peaks:
            key = round(exp_mz, 4)
            count = exp_mz_counts.get(key, 0)
            if count == 0:
                continue
            for theo_mz in theo_mzs:
                if abs(exp_mz - theo_mz) / theo_mz * 1e6 <= frag_ppm:
                    idf += exp_inten * (1.0 / count)
                    break
        return idf

    def _compute_diag_score(self, cfil_subset, fragment_peaks, frag_ppm,
                            skeleton):
        """
        计算诊断碎片得分: 匹配该骨架专属碎片的实验强度之和。

        Parameters
        ----------
        cfil_subset : DataFrame
            当前骨架的 CFIL 子集。
        fragment_peaks : np.ndarray
            实验碎片峰。
        frag_ppm : float
            碎片 ppm 容差。
        skeleton : str
            骨架类型名称。

        Returns
        -------
        float
            诊断碎片得分。
        """
        if len(fragment_peaks) == 0:
            return 0.0
        diag = 0.0
        for _, crow in cfil_subset.iterrows():
            theo_mz = crow["Cal.m/z of Frag"]
            annos = str(crow.get("Annotations", ""))
            matched_inten = 0.0
            for exp_mz, exp_inten in fragment_peaks:
                if abs(exp_mz - theo_mz) / theo_mz * 1e6 <= frag_ppm:
                    matched_inten = exp_inten
                    break
            if matched_inten > 0:
                for a in annos.split("/"):
                    a = a.strip()
                    for diag_anno, diag_skels in self._DIAGNOSTIC_EXCLUSIVE.items():
                        if diag_anno in a and skeleton in diag_skels:
                            diag += matched_inten
                            break
        return diag

    def _select_best_skeleton(self, compound_id, candidates_df,
                              fragment_peaks, fragment_ppm):
        """
        多级评分选择最佳骨架候选。

        策略:
          1. 对每个骨架候选计算 total / idf / diag 三个指标
          2. 分级筛选:
             a. Total 在最高值 95% 以内
             b. 在 a 中 IDF 在最高值 95% 以内
             c. 在 b 中 diag 最高的候选（若 diag>0），否则 total 最高

        Parameters
        ----------
        compound_id : str
            化合物 ID。
        candidates_df : DataFrame
            TPIL 筛选候选（可能含多骨架）。
        fragment_peaks : np.ndarray
            实验碎片峰。
        fragment_ppm : float
            碎片 ppm 容差。

        Returns
        -------
        DataFrame or None
            最佳骨架的 SM 匹配结果。
        """
        if len(candidates_df) == 0:
            return None

        # Step 1: 对每个候选运行 match_single_compound，收集指标
        scored = []  # list of (row, result, total, idf, diag)
        cfil_mz_sets = []  # [(skeleton, set(theo_mzs)] for IDF

        for _, row in candidates_df.iterrows():
            skel = row.get("Skeleton", None)
            if skel is not None and pd.isna(skel):
                skel = None

            result = self.match_single_compound(
                compound_id=compound_id,
                precursor_mz=row.get("Exp.m/z", row.get("cal.m.z.H", 0)),
                fragment_peaks=fragment_peaks,
                precursor_info=row,
                fragment_ppm=fragment_ppm,
            )
            if len(result) == 0:
                continue

            total = result["sum_abund"].sum()

            # 收集 CFIL 子集用于 IDF/diag 计算
            oh = int(row.get("OHsum", 0))
            och3 = int(row.get("OCH3sum", 0))
            cfil_subset = self.cfil[
                (self.cfil["OHsum"] == oh)
                & (self.cfil["OCH3sum"] == och3)
            ].copy()
            if skel is not None:
                cfil_subset = cfil_subset[
                    cfil_subset["Substitution modes"].apply(
                        lambda sm: self._skeleton_matches(sm, skel)
                    )
                ]

            theo_mzs = set(cfil_subset["Cal.m/z of Frag"].round(4))
            cfil_mz_sets.append((skel, theo_mzs))

            scored.append({
                "row": row, "result": result, "total": total,
                "skel": skel, "cfil_subset": cfil_subset,
            })

        if len(scored) == 0:
            return None

        # Step 2: 预计算每个实验 m/z 被多少骨架匹配 (用于 IDF)
        exp_mz_counts = {}
        if len(fragment_peaks) > 0:
            for exp_mz, _ in fragment_peaks:
                key = round(exp_mz, 4)
                count = 0
                for skel, theo_mzs in cfil_mz_sets:
                    for theo_mz in theo_mzs:
                        if abs(exp_mz - theo_mz) / theo_mz * 1e6 <= fragment_ppm:
                            count += 1
                            break
                if count > 0:
                    exp_mz_counts[key] = count

        # Step 3: 计算 IDF 和 diag
        for item in scored:
            item["idf"] = self._compute_idf_score(
                item["cfil_subset"], fragment_peaks, fragment_ppm, exp_mz_counts
            )
            item["diag"] = self._compute_diag_score(
                item["cfil_subset"], fragment_peaks, fragment_ppm,
                item["skel"] or ""
            )

        # Step 4: 分级筛选
        # Level 1: Total 在最高值 75% 以内（宽阈值，避免错过正确骨架）
        max_total = max(item["total"] for item in scored)
        level1 = [i for i in scored if i["total"] >= max_total * 0.75]

        if len(level1) == 1:
            return level1[0]["result"]

        # Level 2: IDF 在最高值 90% 以内
        max_idf = max(i["idf"] for i in level1)
        level2 = [i for i in level1 if i["idf"] >= max_idf * 0.90]

        if len(level2) == 1:
            return level2[0]["result"]

        # Level 3: diag 最高的候选（若 diag 占 total 10% 以上），否则 total 最高
        # 10% 阈值避免弱谱图中巧合匹配的小诊断碎片误导选择
        max_diag = max(i["diag"] for i in level2)
        if max_diag > 0:
            best_diag = max(level2, key=lambda i: i["diag"])
            if best_diag["total"] > 0 and best_diag["diag"] / best_diag["total"] >= 0.10:
                best = best_diag
            else:
                best = max(level2, key=lambda i: i["total"])
        else:
            best = max(level2, key=lambda i: i["total"])

        return best["result"]

    # ==================== 完整运行流程 ====================

    def run_from_spectra(
        self,
        spectra,
        screened_df=None,
        precursor_ppm=10,
        fragment_ppm=50,
    ):
        """
        从谱图列表和筛选结果运行完整的 SM 注释。

        流程:
          1. 如果未提供 screened_df，先用 TPIL 筛选
          2. 对筛选结果中的每个化合物 ID，提取对应谱图的碎片
          3. 对每组 (OHsum, OCH3sum) 组合运行 match_single_compound
          4. 合并所有结果

        Parameters
        ----------
        spectra : list[dict]
            read_mgf() 返回的谱图列表。
        screened_df : pd.DataFrame or None
            CompoundScreener 的筛选结果。如果为 None，需先提供 TPIL。
        precursor_ppm : float
            前体离子 ppm 容差（仅当 screened_df 为 None 时使用）。
        fragment_ppm : float
            碎片离子 ppm 容差。

        Returns
        -------
        pd.DataFrame
            所有化合物的 SM 注释结果。
        """
        # Step 1: 如果没有筛选结果，先做筛选
        if screened_df is None:
            if not hasattr(self, "screener"):
                raise ValueError(
                    "未提供 screened_df，且未加载 TPIL。"
                    "请提供 screened_df 或在初始化时提供 tpil_path。"
                )
            from .mgf_parser import read_mgf
            from .screener import CompoundScreener

            screened_df = self.screener.screen_mgf(
                spectra, precursor_ppm=precursor_ppm
            )

        if len(screened_df) == 0:
            return pd.DataFrame()

        # Step 2: 构建 谱图ID → 碎片峰 的映射
        spec_map = {}
        for idx, spec in enumerate(spectra):
            title = spec.get("title", f"X{idx + 1}")
            if not title:
                title = f"X{idx + 1}"
            spec_map[title] = spec["peaks"]

        # Step 3: 对每个化合物/每组 (OH, OCH3) 做 SM 匹配
        all_results = []

        # 原版 4 类骨架
        ORIGINAL_SKELETONS = {
            "Flavones", "IsoFlavones", "Flavonols (C-3OH)", "Flavonols (C-3OCH3)"
        }

        # 按 Compound ID 分组
        for compound_id, group in screened_df.groupby("Compound ID"):
            if compound_id not in spec_map:
                continue
            fragment_peaks = spec_map[compound_id]

            # --- 骨架候选选择 ---
            # 同一化合物可能有多个 TPIL 筛选候选（不同骨架的异构体）。
            # 扩展版覆盖 16 类骨架，同一质量可能匹配多种骨架类型。
            #
            # 策略: 多级评分选择最佳骨架
            #   1. 对每个骨架候选运行 match_single_compound，计算:
            #      - total: 所有 SM 的 sum_abund 之和
            #      - idf: IDF 加权得分（碎片被越多骨架共享，权重越低）
            #      - diag: 诊断碎片得分（该骨架特有的碎片匹配强度）
            #   2. 分级筛选:
            #      a. Total 在最高值 95% 以内的候选
            #      b. 在 a 中 IDF 在最高值 95% 以内的候选
            #      c. 在 b 中 diag 最高的候选（若有），否则 total 最高的
            #   3. 每个骨架只匹配自己骨架的 SM (骨架一致性过滤)
            has_skeleton = "Skeleton" in group.columns
            if has_skeleton:
                skeleton_valid = group[group["Skeleton"].notna()]
                skeleton_nan = group[group["Skeleton"].isna()]

                best_result = self._select_best_skeleton(
                    compound_id, skeleton_valid, fragment_peaks, fragment_ppm
                )

                # 如果有 Skeleton 的候选都无匹配，用无 Skeleton 的候选
                if best_result is None and len(skeleton_nan) > 0:
                    best_result = self._select_best_skeleton(
                        compound_id, skeleton_nan, fragment_peaks, fragment_ppm
                    )

                if best_result is not None:
                    all_results.append(best_result)
            else:
                # 无 Skeleton 列: 对所有候选运行 SM，取最佳
                best_result = self._select_best_skeleton(
                    compound_id, group, fragment_peaks, fragment_ppm
                )
                if best_result is not None:
                    all_results.append(best_result)

        if len(all_results) == 0:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)
        # 按 sum_abund 降序排列
        combined = combined.sort_values(
            ["Compound ID", "sum_abund"], ascending=[True, False]
        ).reset_index(drop=True)
        return combined

    def run(self, mgf_path, precursor_ppm=10, fragment_ppm=50):
        """
        便捷方法：直接从 MGF 文件运行完整的 SM 注释流程。

        Parameters
        ----------
        mgf_path : str or Path
            MGF 文件路径。
        precursor_ppm : float
            前体离子 ppm 容差。
        fragment_ppm : float
            碎片离子 ppm 容差。

        Returns
        -------
        pd.DataFrame
            SM 注释结果。
        """
        from .mgf_parser import read_mgf

        spectra = read_mgf(mgf_path)
        return self.run_from_spectra(
            spectra,
            screened_df=None,
            precursor_ppm=precursor_ppm,
            fragment_ppm=fragment_ppm,
        )

    def save_results(self, results, output_path):
        """
        将结果保存为 Excel 文件。

        Parameters
        ----------
        results : pd.DataFrame
            run() 或 run_from_spectra() 返回的结果。
        output_path : str or Path
            输出文件路径（.xlsx）。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_excel(output_path, index=False)
        return output_path
