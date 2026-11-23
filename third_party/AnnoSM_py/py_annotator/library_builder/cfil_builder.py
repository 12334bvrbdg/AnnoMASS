"""
CFIL (Characteristic Fragment Ion Library) 构建器
===================================================

根据 fragmentation_rules.py 中定义的裂解规则，为每种黄酮骨架类型
× OH/OCH3 取代基组合生成特征碎片离子库。

输出列与原版 CFIL 一致:
  Substitution modes, Class, Types and positions of substituents,
  OHsum, OCH3sum, Formulas of Frag, Cal.m/z of Frag, Annotations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

from .fragmentation_rules import (
    ALL_RULES,
    ORIGINAL_CLASSES,
    A_RING_BASE,
    B_RING_BASE,
    NEUTRAL_LOSS,
    MASS,
    calc_substituted_mass,
    apply_neutral_losses,
)


class CFILBuilder:
    """
    根据裂解规则构建扩展版 CFIL。

    Parameters
    ----------
    max_oh_per_ring : int
        每个环最多考虑的 OH 数量，默认 5。
    max_och3_per_ring : int
        每个环最多考虑的 OCH3 数量，默认 5。
    """

    def __init__(self, max_oh_per_ring=5, max_och3_per_ring=5):
        self.max_oh = max_oh_per_ring
        self.max_och3 = max_och3_per_ring

    def _generate_substitution_modes(self, skeleton_class):
        """
        为指定骨架类型生成所有可能的 Substitution modes 组合。

        SM 格式: <prefix>A<OH><OCH3>B<OH><OCH3>C<OH><OCH3>
        其中 C 环编码固定为骨架特征。

        Returns
        -------
        list[dict]
            每个元素包含: sm_name, a_oh, a_och3, b_oh, b_och3,
                          c_oh, c_och3, oh_sum, och3_sum
        """
        rules = ALL_RULES.get(skeleton_class, {})
        prefix = rules.get("prefix", "")
        c_code = rules.get("c_ring_code", "00")

        # C 环的 OH/OCH3 固定
        c_oh = int(c_code[0])
        c_och3 = int(c_code[1])

        modes = []
        for a_oh, a_och3 in product(range(self.max_oh + 1), range(self.max_och3 + 1)):
            for b_oh, b_och3 in product(range(self.max_oh + 1), range(self.max_och3 + 1)):
                # 至少要有一些取代基（跳过全0的组合也保留，因为原版有A00B00C00）
                oh_sum = a_oh + b_oh + c_oh
                och3_sum = a_och3 + b_och3 + c_och3

                # 限制总取代基数量 (避免组合爆炸)
                if oh_sum + och3_sum > 12:
                    continue

                sm_name = f"{prefix}A{a_oh}{a_och3}B{b_oh}{b_och3}C{c_oh}{c_och3}"

                modes.append({
                    "sm_name": sm_name,
                    "a_oh": a_oh,
                    "a_och3": a_och3,
                    "b_oh": b_oh,
                    "b_och3": b_och3,
                    "c_oh": c_oh,
                    "c_och3": c_och3,
                    "oh_sum": oh_sum,
                    "och3_sum": och3_sum,
                })

        return modes

    def _calc_a_ring_fragment(self, rda_channel, a_oh, a_och3, ring_type):
        """
        计算 A 环 RDA 碎片的 m/z。

        Parameters
        ----------
        rda_channel : str
            RDA 通道名称，如 "1,3A+"
        a_oh : int
            A 环 OH 数量。
        a_och3 : int
            A 环 OCH3 数量。
        ring_type : str
            环类型 ("carbonyl" / "saturated" / "oxonium" / "open_chain")

        Returns
        -------
        tuple(float, str) or None
            (m/z, formula_str)，无法计算返回 None。
        """
        key = (ring_type, rda_channel)
        if key not in A_RING_BASE:
            return None

        formula_str, base_mass = A_RING_BASE[key]
        mass = calc_substituted_mass(base_mass, a_oh, a_och3, is_a_ring=True)
        return mass, formula_str

    def _calc_b_ring_fragment(self, rda_channel, b_oh, b_och3):
        """
        计算 B 环 RDA 碎片的 m/z。

        Returns
        -------
        tuple(float, str) or None
            (m/z, formula_str)
        """
        if rda_channel not in B_RING_BASE:
            return None

        formula_str, base_mass = B_RING_BASE[rda_channel]
        mass = calc_substituted_mass(base_mass, b_oh, b_och3, is_a_ring=False)
        return mass, formula_str

    def _generate_loss_ladders(self, base_mass, base_formula, base_annotation,
                                neutral_losses, max_co=2, n_oh=0, n_och3=0):
        """
        生成基础碎片的中性丢失级联。

        生成规则 (参考原版 CFIL):
          - CO 丢失: 1 到 max_co 次
          - H2O 丢失: 仅当有 OH 时, 1 到 min(n_oh, 2) 次
          - CH3 丢失: 仅当有 OCH3 时, 1 到 min(n_och3, 2) 次
          - 组合: CO + H2O, CO + CH3 等

        Parameters
        ----------
        base_mass : float
            基础碎片 m/z。
        base_formula : str
            基础碎片分子式。
        base_annotation : str
            基础碎片注释 (如 "1，3A+")。
        neutral_losses : list[str]
            支持的中性丢失类型。
        max_co : int
            最大 CO 丢失次数。
        n_oh : int
            该碎片来源环的 OH 数 (决定 H2O 丢失)。
        n_och3 : int
            该碎片来源环的 OCH3 数 (决定 CH3 丢失)。

        Returns
        -------
        list[dict]
            丢失级联列表，每个包含 m/z, formula, annotation。
        """
        results = [{
            "mz": base_mass,
            "formula": base_formula,
            "annotation": base_annotation,
        }]

        # CO 丢失级联
        co_count = min(max_co, 2)
        co_losses = []
        for i in range(1, co_count + 1):
            loss_str = f"{i}×CO" if i > 1 else "CO"
            new_mass = apply_neutral_losses(base_mass, ["CO"] * i)
            new_annotation = f"{base_annotation}-{loss_str}"
            co_losses.append({
                "mz": new_mass,
                "formula": base_formula,  # 简化: 不精确计算丢失后公式
                "annotation": new_annotation,
            })
        results.extend(co_losses)

        # H2O 丢失 (仅当有 OH)
        if "H2O" in neutral_losses and n_oh > 0:
            max_h2o = min(n_oh, 2)
            for i in range(1, max_h2o + 1):
                loss_str = f"{i}×H2O" if i > 1 else "H2O"
                new_mass = apply_neutral_losses(base_mass, ["H2O"] * i)
                new_annotation = f"{base_annotation}-{loss_str}"
                results.append({
                    "mz": new_mass,
                    "formula": base_formula,
                    "annotation": new_annotation,
                })

                # CO + H2O 组合
                for j in range(1, co_count + 1):
                    co_str = f"{j}×CO" if j > 1 else "CO"
                    combo_mass = apply_neutral_losses(
                        base_mass, ["H2O"] * i + ["CO"] * j
                    )
                    results.append({
                        "mz": combo_mass,
                        "formula": base_formula,
                        "annotation": f"{base_annotation}-{loss_str}-{co_str}",
                    })

        # CH3 丢失 (仅当有 OCH3)
        if "CH3" in neutral_losses and n_och3 > 0:
            max_ch3 = min(n_och3, 2)
            for i in range(1, max_ch3 + 1):
                loss_str = f"{i}×CH3" if i > 1 else "CH3"
                new_mass = apply_neutral_losses(base_mass, ["CH3"] * i)
                new_annotation = f"{base_annotation}-{loss_str}"
                results.append({
                    "mz": new_mass,
                    "formula": base_formula,
                    "annotation": new_annotation,
                })

                # CO + CH3 组合
                for j in range(1, co_count + 1):
                    co_str = f"{j}×CO" if j > 1 else "CO"
                    combo_mass = apply_neutral_losses(
                        base_mass, ["CH3"] * i + ["CO"] * j
                    )
                    results.append({
                        "mz": combo_mass,
                        "formula": base_formula,
                        "annotation": f"{base_annotation}-{loss_str}-{co_str}",
                    })

        return results

    def _generate_fragments_for_mode(self, mode, skeleton_class):
        """
        为一个 Substitution mode 生成所有特征碎片离子。

        Parameters
        ----------
        mode : dict
            _generate_substitution_modes 返回的 mode 字典。
        skeleton_class : str
            骨架类型名称。

        Returns
        -------
        list[dict]
            碎片离子列表，每个包含 mz, formula, annotation。
        """
        rules = ALL_RULES.get(skeleton_class, {})
        ring_type = rules.get("ring_type", "carbonyl")
        neutral_losses = rules.get("neutral_losses", ["CO", "H2O", "CH3"])
        max_co = rules.get("max_co_losses", 2)

        fragments = []

        # --- RDA 裂解碎片 ---
        for rda_channel, ring_label in rules.get("rda_cleavages", []):
            if ring_label == "A":
                result = self._calc_a_ring_fragment(
                    rda_channel, mode["a_oh"], mode["a_och3"], ring_type
                )
                if result is None:
                    continue
                base_mass, formula = result
                n_oh = mode["a_oh"]
                n_och3 = mode["a_och3"]
            elif ring_label == "B":
                result = self._calc_b_ring_fragment(
                    rda_channel, mode["b_oh"], mode["b_och3"]
                )
                if result is None:
                    continue
                base_mass, formula = result
                n_oh = mode["b_oh"]
                n_och3 = mode["b_och3"]
            else:
                continue

            # 生成丢失级联
            loss_ladders = self._generate_loss_ladders(
                base_mass, formula, rda_channel,
                neutral_losses, max_co, n_oh, n_och3
            )
            fragments.extend(loss_ladders)

        # --- 骨架特有碎片 ---
        for special in rules.get("special_fragments", []):
            if len(special) == 4:
                ann, ring_label, formula, base_mass = special
            elif len(special) == 3:
                ann, ring_label, formula = special
                base_mass = None
            else:
                continue

            if base_mass is None:
                continue

            # 对特殊碎片也应用取代基修正
            if ring_label == "A":
                mass = calc_substituted_mass(
                    base_mass, mode["a_oh"], mode["a_och3"]
                )
                n_oh = mode["a_oh"]
                n_och3 = mode["a_och3"]
            elif ring_label == "B":
                mass = calc_substituted_mass(
                    base_mass, mode["b_oh"], mode["b_och3"]
                )
                n_oh = mode["b_oh"]
                n_och3 = mode["b_och3"]
            else:
                mass = base_mass
                n_oh = mode["oh_sum"]
                n_och3 = mode["och3_sum"]

            loss_ladders = self._generate_loss_ladders(
                mass, formula, ann,
                neutral_losses, max_co, n_oh, n_och3
            )
            fragments.extend(loss_ladders)

        return fragments

    def build(self, skeleton_classes=None):
        """
        构建扩展版 CFIL。

        Parameters
        ----------
        skeleton_classes : list[str] or None
            要构建的骨架类型列表。None 表示全部 16 种。

        Returns
        -------
        pd.DataFrame
            CFIL 数据，列与原版一致。
        """
        if skeleton_classes is None:
            skeleton_classes = list(ALL_RULES.keys())

        all_rows = []

        for skel_class in skeleton_classes:
            if skel_class not in ALL_RULES:
                print(f"  跳过未知骨架类型: {skel_class}")
                continue

            print(f"  构建 {skel_class}...", end=" ")
            modes = self._generate_substitution_modes(skel_class)
            count = 0

            for mode in modes:
                fragments = self._generate_fragments_for_mode(mode, skel_class)

                for frag in fragments:
                    all_rows.append({
                        "Substitution modes": mode["sm_name"],
                        "Class": skel_class,
                        "Types and positions of substituents": (
                            f"{ALL_RULES[skel_class].get('prefix','')}+"
                            f"RingA(OH×{mode['a_oh']}+OCH3×{mode['a_och3']})+"
                            f"RingB(OH×{mode['b_oh']}+OCH3×{mode['b_och3']})+"
                            f"RingC(OH×{mode['c_oh']}+OCH3×{mode['c_och3']})"
                        ),
                        "OHsum": mode["oh_sum"],
                        "OCH3sum": mode["och3_sum"],
                        "Formulas of Frag": frag["formula"],
                        "Cal.m/z of Frag": frag["mz"],
                        "Annotations": frag["annotation"],
                    })
                    count += 1

            print(f"{len(modes)} modes, {count} fragments")

        cfil_df = pd.DataFrame(all_rows)
        print(f"\nCFIL 构建完成: {len(cfil_df)} 行, "
              f"{cfil_df['Substitution modes'].nunique()} 种 SM")

        return cfil_df

    def merge_with_original(self, new_cfil_df, original_cfil_path=None):
        """
        将新生成的 CFIL 与原版 CFIL 合并。

        Parameters
        ----------
        new_cfil_df : pd.DataFrame
            新生成的 CFIL。
        original_cfil_path : str or Path, optional
            原版 CFIL.xlsx 路径。

        Returns
        -------
        pd.DataFrame
            合并后的 CFIL。
        """
        if original_cfil_path is None:
            original_cfil_path = Path(__file__).parent.parent.parent / "Library" / "CFIL.xlsx"
        else:
            original_cfil_path = Path(original_cfil_path)

        if original_cfil_path.exists():
            original = pd.read_excel(original_cfil_path)
            print(f"合并原版 CFIL: {len(original)} 行")

            # 确保列一致
            for col in original.columns:
                if col not in new_cfil_df.columns:
                    new_cfil_df[col] = None

            combined = pd.concat([original, new_cfil_df[original.columns]], ignore_index=True)
            return combined
        else:
            print(f"原版 CFIL 不存在: {original_cfil_path}，仅使用新生成的数据")
            return new_cfil_df

    def save(self, cfil_df, output_path):
        """
        保存 CFIL 为文件。
        超过 Excel 行数限制(1048576)时自动转为 CSV。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(cfil_df) > 1048000:
            # 超过 Excel 限制，改用 CSV
            csv_path = output_path.with_suffix(".csv")
            cfil_df.to_csv(csv_path, index=False)
            print(f"CFIL 已保存(CSV, 因超过Excel行数限制): {csv_path} ({len(cfil_df)} 行)")
            return csv_path

        if output_path.suffix.lower() == ".csv":
            cfil_df.to_csv(output_path, index=False)
        else:
            cfil_df.to_excel(output_path, index=False)
        print(f"CFIL 已保存: {output_path} ({len(cfil_df)} 行)")
        print(f"CFIL 已保存: {output_path} ({len(cfil_df)} 行)")
        return output_path
