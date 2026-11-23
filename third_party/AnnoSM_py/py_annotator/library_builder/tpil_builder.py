"""
TPIL (Precursor Ion Library) 构建器
====================================

从 DeepMASS2 黄酮结构数据库读取 SMILES，用 RDKit 计算精确质量、
统计 OH/OCH3 数量，按骨架类型分组生成扩展版 TPIL。

输出列与原版 TPIL 一致:
  Compound group, Class, OHsum, OCH3sum,
  Sugar and phenylpropionyl units, S&Psum, Formulas, Exact Mass, cal.m.z.H
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .fragmentation_rules import ALL_RULES, ORIGINAL_CLASSES


# DeepMASS2 的 flavclass_filter 中骨架名称到 TPIL Class 的映射
# flavclass_filter.classify() 返回的集合可能包含多个类，取优先级最高的
# 注意: 优先级经过优化，确保每个化合物分到最特异的骨架类型
#  - Flavanone 优先于 Dihydrochalcone (Liquiritigenin/Naringenin 问题:
#    两者SMARTS都匹配, 但黄烷酮有C环结构更特异, 应优先)
#  - Flavan3ol 优先于 Flavanone (Catechin 不应被分到 Flavanone)
#  - Pterocarpan/IsoflavanChroman 在最前 (四环/三环特异结构)
CLASS_PRIORITY = [
    "Pterocarpan",        # 紫檀烷: 最特异的四环结构,优先匹配
    "IsoflavanChroman",   # 异黄烷色满: 三环结构 (Isomucronulatol)
    "Aurone",             # 橙酮: 五元环,特异
    "Anthocyanidin",      # 花青素: 氧鎓离子
    "Chalcone",           # 查尔酮: 开链但不饱和
    "Isoflavan",          # 异黄烷: C环全饱和,B环在C3
    "Isoflavanone",       # 异黄烷酮: C环2,3饱和
    "Flavan3ol",          # 黄烷醇: 无C=O,有3-OH (优先于Flavanone)
    "Flavanonol",         # 二氢黄酮醇: C环饱和+3-OH
    "Flavanone",          # 黄烷酮: C环2,3饱和 (优先于Dihydrochalcone!)
    "Dihydrochalcone",    # 二氢查尔酮: 开链饱和 (后于Flavanone)
    "Flavonol",           # 黄酮醇: 3-OH
    "Flavone",            # 黄酮
    "Isoflavone",         # 异黄酮
    "Flavan",             # 黄烷
    "Xanthone",
    "ChromoneCore",
]

# flavclass_filter 名称 -> TPIL/CFIL 中的 Class 名称
CLASS_NAME_MAP = {
    "Flavone": "Flavones",
    "Flavonol": "Flavonols (C-3OH)",   # 默认归为3-OH黄酮醇
    "Flavanone": "Flavanone",
    "Flavanonol": "Flavanonol",
    "Flavan3ol": "Flavan3ol",
    "Isoflavone": "IsoFlavones",
    "Isoflavanone": "Isoflavanone",
    "Isoflavan": "Isoflavan",
    "Chalcone": "Chalcone",
    "Dihydrochalcone": "Dihydrochalcone",
    "Aurone": "Aurone",
    "Anthocyanidin": "Anthocyanidin",
    "Flavan": "Flavan",
    "Pterocarpan": "Pterocarpan",
    "Xanthone": "Xanthone",
    "IsoflavanChroman": "Isoflavan",   # 归入异黄烷
    "ChromoneCore": None,              # 太泛化,跳过
}


def _infer_skeleton_from_class(class_value):
    """
    根据原版 TPIL 的 Class 列推断骨架类型。
    原版 TPIL 的 Class 只有 "Aglycone" 和 "Glycosylated flavonoids"，
    无法区分骨架类型，返回 None。
    但如果原版 TPIL 有更详细的 Class 信息，则尝试映射。
    """
    if class_value is None or pd.isna(class_value):
        return None
    cv = str(class_value).strip()
    # 原版 TPIL 的 Class 只有 Aglycone / Glycosylated flavonoids
    # 无法区分骨架，返回 None 让 SM 注释不做骨架过滤
    return None


# 原版 TPIL 的糖基单元列表 (用于生成糖苷化黄酮)
SUGAR_UNITS = [
    ("-", 0),
    ("Hexose (Hex, -C6H10O5: Glc/Gal/Man)", 1),
    ("deoxyhexose (dHex, -C6H10O4: Rha)", 1),
    ("pentose (Pen, -C5H8O4: Ara/Xyl)", 1),
    ("hexose acid (HexA, -C6H8O6: GluA/GalA)", 1),
]


class TPILBuilder:
    """
    从 DeepMASS2 结构数据库构建扩展版 TPIL。

    Parameters
    ----------
    structure_db_path : str or Path
        DeepMASS2 的 FlavonoidStructureDB-optimized.csv 路径。
    flavclass_module_path : str or Path, optional
        DeepMASS2 的 core 目录路径（用于 import flavclass_filter）。
        如果为 None，则尝试从 structure_db_path 推断。
    """

    def __init__(self, structure_db_path, flavclass_module_path=None):
        self.structure_db_path = Path(structure_db_path)
        if not self.structure_db_path.exists():
            raise FileNotFoundError(f"结构数据库不存在: {self.structure_db_path}")

        # 导入 DeepMASS2 的 flavclass_filter
        self._import_flavclass(flavclass_module_path)

    def _import_flavclass(self, module_path):
        """导入 DeepMASS2 的 flavclass_filter 模块"""
        if module_path is None:
            # 从结构数据库路径推断
            # structure_db_path = .../Annomass/Stru/...
            parent = self.structure_db_path.parent.parent
            core_path = parent / "core"
            if core_path.exists():
                module_path = core_path
            else:
                raise ImportError(
                    "无法自动找到 DeepMASS2 的 core 目录。"
                    "请显式提供 flavclass_module_path 参数。"
                )

        module_path = Path(module_path)
        if str(module_path) not in sys.path:
            sys.path.insert(0, str(module_path.parent))

        try:
            from core.flavclass_filter import classify, is_flavonoid
            self._classify = classify
            self._is_flavonoid = is_flavonoid
        except ImportError as e:
            raise ImportError(
                f"无法导入 DeepMASS2 的 flavclass_filter: {e}\n"
                f"请确认 core 目录路径正确: {module_path}"
            )

    def _count_oh_och3(self, mol):
        """
        统计分子中的 OH 和 OCH3 数量。

        OH: -OH (羟基氧, 度数2: 连1个C + 1个H)
        OCH3: -OCH3 (甲氧基氧, 度数2: 连1个C + 1个CH3)

        注意: SMARTS 中 X<n> 指的是总连接度 (含隐式H)。
        - 羟基氧 O-H 连1个C, 隐式H=1, 总度数=2 -> [OX2H1]
        - 羰基氧 O=C 连1个C, 隐式H=0, 总度数=1 -> [OX1H0]
        - 醚氧 O-C-C 连2个C, 隐式H=0, 总度数=2 -> [OX2H0]
        - 甲氧基氧 O-CH3 连1个C + 1个CH3, 总度数=2 -> [OX2H0]

        Returns
        -------
        tuple(int, int)
            (OH数量, OCH3数量)
        """
        if mol is None:
            return 0, 0

        # [OX2H1]: 羟基氧 (连接1个原子 + 1个隐式H)
        oh_patt = Chem.MolFromSmarts("[OX2H1]")
        # [OX2H0][CH3]: 甲氧基氧 (连接1个非H原子 + 1个CH3, 无隐式H)
        och3_patt = Chem.MolFromSmarts("[OX2H0][CH3]")

        oh_count = len(mol.GetSubstructMatches(oh_patt)) if oh_patt else 0
        och3_count = len(mol.GetSubstructMatches(och3_patt)) if och3_patt else 0

        return oh_count, och3_count

    def _calc_exact_mass(self, mol):
        """计算分子的单同位素精确质量"""
        if mol is None:
            return None
        return round(Descriptors.ExactMolWt(mol), 4)

    def _formula_to_str(self, mol):
        """获取分子式字符串 (如 'C15 H10 O4')"""
        if mol is None:
            return ""
        formula = rdMolDescriptors.CalcMolFormula(mol)
        # 转换为原版格式: C15 H10 O4
        return formula

    def _classify_skeleton(self, smiles):
        """
        对 SMILES 进行骨架分类。

        Returns
        -------
        str or None
            骨架类型名称 (如 "Flavanone")，无法分类返回 None。
        """
        classes = self._classify(smiles)

        # 按优先级选择
        for cls in CLASS_PRIORITY:
            if cls in classes:
                tpil_class = CLASS_NAME_MAP.get(cls)
                return tpil_class

        return None

    def build(self, aglycone_only=True, max_compounds=50000):
        """
        构建扩展版 TPIL。

        Parameters
        ----------
        aglycone_only : bool
            如果 True，只保留苷元（无糖基）的黄酮。
            如果 False，也生成糖苷化黄酮的 TPIL 行。
        max_compounds : int
            最大处理化合物数（防止内存溢出）。

        Returns
        -------
        pd.DataFrame
            TPIL 数据，列与原版一致。
        """
        # 读取结构数据库
        df = pd.read_csv(self.structure_db_path)
        print(f"读取结构数据库: {len(df)} 个化合物")

        results = []
        compound_group = 1

        for idx, row in df.head(max_compounds).iterrows():
            smiles = row.get("SMILES", "")
            if not smiles or pd.isna(smiles):
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 分类骨架
            skeleton = self._classify_skeleton(smiles)
            if skeleton is None:
                continue

            # 统计 OH/OCH3
            oh_sum, och3_sum = self._count_oh_och3(mol)

            # 计算精确质量
            exact_mass = self._calc_exact_mass(mol)
            if exact_mass is None:
                continue

            # [M+H]+ m/z
            cal_mz_h = round(exact_mass + 1.00728, 4)

            # 分子式
            formula = self._formula_to_str(mol)

            results.append({
                "Compound group": compound_group,
                "Class": "Aglycone",
                "OHsum": oh_sum,
                "OCH3sum": och3_sum,
                "Sugar and phenylpropionyl units": "-",
                "S&Psum": 0,
                "Formulas": formula,
                "Exact Mass": exact_mass,
                "cal.m.z.H": cal_mz_h,
                "_skeleton": skeleton,  # 内部字段，后续移除
            })
            compound_group += 1

        tpil_df = pd.DataFrame(results)
        print(f"分类完成: {len(tpil_df)} 个黄酮化合物")

        # 按骨架类型统计
        if "_skeleton" in tpil_df.columns:
            print("\n骨架类型分布:")
            for skel, count in tpil_df["_skeleton"].value_counts().items():
                print(f"  {skel}: {count}")

        return tpil_df

    def build_summary(self, tpil_df):
        """
        从详细 TPIL 生成骨架类型 × OHsum × OCH3sum × ExactMass 的汇总表。
        每个 (骨架, OH, OCH3, 精确质量) 组合只保留一条记录（去重）。

        注意: 同一骨架类型 + OH/OCH3 组合可能有多个不同质量的化合物
        (如 Flavanone/OH=2/OCH3=0 有 pinocembrin C15H12O4 和
        2,5,7-trihydroxyflavanone C15H12O5)，需要全部保留。

        Parameters
        ----------
        tpil_df : pd.DataFrame
            build() 返回的完整 TPIL。

        Returns
        -------
        pd.DataFrame
            去重后的 TPIL（每个骨架×OH×OCH3×质量组合一条）。
            保留 Skeleton 列用于后续骨架一致性过滤。
        """
        if len(tpil_df) == 0:
            return tpil_df

        # 按 (骨架, OHsum, OCH3sum, Exact Mass) 去重
        summary = tpil_df.drop_duplicates(
            subset=["_skeleton", "OHsum", "OCH3sum", "Exact Mass"], keep="first"
        ).copy()

        # 重命名 _skeleton -> Skeleton (保留用于骨架一致性过滤)
        summary = summary.rename(columns={"_skeleton": "Skeleton"})

        # 重新编号
        summary["Compound group"] = range(1, len(summary) + 1)

        return summary

    def merge_with_original(self, new_tpil_df, original_tpil_path=None):
        """
        将新生成的 TPIL 与原版 TPIL 合并。

        Parameters
        ----------
        new_tpil_df : pd.DataFrame
            新生成的 TPIL。
        original_tpil_path : str or Path, optional
            原版 TPIL.xlsx 路径。如果为 None，使用默认路径。

        Returns
        -------
        pd.DataFrame
            合并后的 TPIL。
        """
        if original_tpil_path is None:
            original_tpil_path = Path(__file__).parent.parent.parent / "Library" / "TPIL.xlsx"
        else:
            original_tpil_path = Path(original_tpil_path)

        if original_tpil_path.exists():
            original = pd.read_excel(original_tpil_path)
            original.columns = [c.strip() for c in original.columns]
            print(f"合并原版 TPIL: {len(original)} 行")

            # 清理内部字段
            if "_skeleton" in new_tpil_df.columns:
                new_tpil_df = new_tpil_df.drop(columns=["_skeleton"])

            # 原版 TPIL 没有 Skeleton 列，为其补充（根据 Class 推断）
            if "Skeleton" not in original.columns:
                original["Skeleton"] = original["Class"].map(
                    lambda c: _infer_skeleton_from_class(c)
                )

            # 确保列对齐
            for col in new_tpil_df.columns:
                if col not in original.columns:
                    original[col] = None
            for col in original.columns:
                if col not in new_tpil_df.columns:
                    new_tpil_df[col] = None

            combined = pd.concat(
                [original[new_tpil_df.columns], new_tpil_df], ignore_index=True
            )
            combined["Compound group"] = range(1, len(combined) + 1)
            return combined
        else:
            print(f"原版 TPIL 不存在: {original_tpil_path}，仅使用新生成的数据")
            if "_skeleton" in new_tpil_df.columns:
                new_tpil_df = new_tpil_df.drop(columns=["_skeleton"])
            return new_tpil_df

    def save(self, tpil_df, output_path):
        """保存 TPIL 为 Excel 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tpil_df.to_excel(output_path, index=False)
        print(f"TPIL 已保存: {output_path} ({len(tpil_df)} 行)")
        return output_path
