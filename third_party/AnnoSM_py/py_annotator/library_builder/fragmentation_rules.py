"""
黄酮骨架裂解规则定义
=====================

参考原版 AnnoSM CFIL 的裂解规则，为 12 种新增黄酮骨架类型定义特征碎片离子规则。

原版 CFIL 覆盖的 4 类:
  - Flavones: RDA 裂解 (1,3A+ / 1,4A+ / 0,2B+ / 1,3B+ / 0,4B+ / 0,4A+)
  - IsoFlavones: RDA + [B ring]+ + [M+H-B ring]+
  - Flavonols (C-3OH): RDA + 0,2A+ + 1,2B+
  - Flavonols (C-3OCH3): RDA + 0,2A+

SM 命名规则: <prefix>A<OH><OCH3>B<OH><OCH3>C<OH><OCH3>
  其中 C 环编码固定为骨架特征:
    Flavones/IsoFlavones/Flavanone/...  -> C00
    Flavonols (C-3OH)                   -> C10
    Flavonols (C-3OCH3)                 -> C01

中性丢失质量:
  CO=27.9949  H2O=18.0106  CH3=15.0235  CH4=16.0313
  C2H6=30.0470  CH3OH=32.0262
"""

# ====================================================================
# 原子/碎片质量 (单同位素质量)
# ====================================================================
MASS = {
    "C": 12.0000,
    "H": 1.00783,
    "O": 15.9949,
    "N": 14.0031,
    "proton": 1.00728,  # [M+H]+ 的 H
}

# 中性丢失质量 (精确)
NEUTRAL_LOSS = {
    "CO": 27.9949,
    "H2O": 18.0106,
    "CH3": 15.0235,      # CH3 radical
    "CH4": 16.0313,
    "C2H6": 30.0470,
    "CH3OH": 32.0262,
    "C3H9": 44.0626,     # 3×CH3
    "CO2": 43.9898,
}

# ====================================================================
# 原版 4 类 (用于参考和合并)
# ====================================================================
ORIGINAL_CLASSES = [
    "Flavones",
    "IsoFlavones",
    "Flavonols (C-3OH)",
    "Flavonols (C-3OCH3)",
]

# ====================================================================
# A 环基础碎片公式
# ====================================================================
# 黄酮的 A 环部分 (含 C2=O 羰基的 RDA 碎片)
# 对于不同骨架，A 环碎片的碳骨架略有不同
#
# 原版 Flavones 的 A 环 RDA 碎片:
#   1,3A+ = C7H5O2+ (无取代, m/z 121.0284)
#   每个 A-OH 增加 O (15.9949)
#   每个 A-OCH3 增加 CH2 (14.0157, 因为 OCH3 = OH + CH2)
#
# B 环 RDA 碎片:
#   0,2B+ = C7H5O+ (无取代, m/z 105.0335)  [Flavones]
#   1,3B+ = C8H7O+ (无取代)
#   每个 B-OH 增加 O
#   每个 B-OCH3 增加 CH2

# A 环基础碎片 (无取代时的公式和 m/z)
# key: (骨架类型, RDA通道) -> (formula_str, base_mass)
A_RING_BASE = {
    # Flavone/Isoflavone/Flavanone/Flavanonol 等含 C=O 的骨架
    # 1,3A+: C7H5O2+ (保留 A环 + C4=O)
    ("carbonyl", "1,3A+"): ("C7H5O2", 121.0284),
    # 1,4A+: C7H5O+ (保留 A环, 不含 C4=O)
    ("carbonyl", "1,4A+"): ("C7H5O", 105.0335),
    # 0,4A+: C6H5O+ (A环少一个碳)
    ("carbonyl", "0,4A+"): ("C6H5O", 93.0335),
    # 0,3A+: C6H5+ (A环纯芳烃)
    ("carbonyl", "0,3A+"): ("C6H5", 77.0386),
    # 0,2A+: C8H5O3+ (含 C2=O + C3, 仅3-取代黄酮醇)
    ("carbonyl", "0,2A+"): ("C8H5O3", 149.0233),

    # Flavan3ol/Flavan 等无 C=O 的骨架 (C环饱和)
    # 1,3A+: C7H7O+ (多2个H, 因为C环饱和)
    ("saturated", "1,3A+"): ("C7H7O", 107.0491),
    # 1,4A+: C7H7+ (无氧)
    ("saturated", "1,4A+"): ("C7H7", 91.0542),
    # 0,4A+: C6H7+ (A环少一碳, 饱和)
    ("saturated", "0,4A+"): ("C6H7", 79.0542),
}

# B 环基础碎片
B_RING_BASE = {
    # 0,2B+: C7H5O+ (标准 B 环 RDA)
    "0,2B+": ("C7H5O", 105.0335),
    # 1,3B+: C8H7O+ (多一个碳)
    "1,3B+": ("C8H7O", 107.0491),
    # 0,4B+: C9H7O+ (完整 B 环 + 桥碳)
    "0,4B+": ("C9H7O", 131.0491),
    # 0,3B+: C8H7+ (无氧)
    "0,3B+": ("C8H7", 91.0542),
    # 1,2B+: C7H7O+ (仅3-取代黄酮醇)
    "1,2B+": ("C7H7O", 107.0491),
    # 1,4B+: C8H7O+ (异黄酮特有)
    "1,4B+": ("C8H7O", 107.0491),
    # [B ring]+: 完整 B 环正离子
    "[B ring]+": ("C6H5", 77.0386),
}


def calc_substituted_mass(base_mass, n_oh, n_och3, is_a_ring=True):
    """
    根据取代基数量计算碎片质量。

    每个 OH  -> +O (15.9949)
    每个 OCH3 -> +O + CH2 (15.9949 + 14.0157 = 30.0106)

    Parameters
    ----------
    base_mass : float
        无取代时的基础碎片质量。
    n_oh : int
        该环上的 OH 数量。
    n_och3 : int
        该环上的 OCH3 数量。
    is_a_ring : bool
        是否 A 环（影响 H 原子数修正，但质量差异可忽略）。

    Returns
    -------
    float
        取代后的碎片质量。
    """
    mass = base_mass
    mass += n_oh * MASS["O"]
    mass += n_och3 * (MASS["O"] + 2 * MASS["H"] + MASS["C"])  # OCH3 = O + CH2
    return round(mass, 4)


def apply_neutral_losses(base_mass, losses):
    """
    对基础碎片质量应用中性丢失链。

    Parameters
    ----------
    base_mass : float
        基础碎片质量。
    losses : list[str]
        中性丢失列表，如 ["CO", "H2O"] 表示先丢失 CO 再丢失 H2O。

    Returns
    -------
    float
        丢失后的质量。
    """
    mass = base_mass
    for loss in losses:
        if loss in NEUTRAL_LOSS:
            mass -= NEUTRAL_LOSS[loss]
    return round(mass, 4)


# ====================================================================
# 12 种新增骨架类型的裂解规则
# ====================================================================
SKELETON_RULES = {
    # ----------------------------------------------------------------
    # 1. Flavanone (黄烷酮) - 如 Naringenin, Liquiritigenin
    #    C环 2,3 位饱和, 有 C4=O
    #    RDA 裂解与 Flavone 类似但碎片多 2H
    # ----------------------------------------------------------------
    "Flavanone": {
        "prefix": "Fla",
        "c_ring_code": "00",
        "ring_type": "carbonyl",       # 含 C4=O
        "hydrogen_delta": 2,           # C环饱和多2个H
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [
            # B环直接断裂 (Flavanone也会产生[M+H-B ring]+碎片)
            ("[B ring]+", "B", "C6H5", 77.0386),
            ("[M+H-B ring]+", "AC", "C9H7O2", 147.0441),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 2. Flavanonol / Dihydroflavonol (二氢黄酮醇) - 如 Taxifolin
    #    C环饱和 + 3-OH
    # ----------------------------------------------------------------
    "Flavanonol": {
        "prefix": "Fno",
        "c_ring_code": "10",           # C环有1个OH (3-OH)
        "ring_type": "carbonyl",
        "hydrogen_delta": 2,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2A+", "A"),            # 3-OH 特有
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("1,2B+", "B"),            # 3-OH 特有
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 3. Flavan3ol (黄烷醇) - 如 Catechin, Epicatechin
    #    C环饱和, 无 C4=O, 有 3-OH
    # ----------------------------------------------------------------
    "Flavan3ol": {
        "prefix": "F3o",
        "c_ring_code": "10",           # C环有1个OH (3-OH)
        "ring_type": "saturated",      # 无 C=O
        "hydrogen_delta": 4,           # C环完全饱和多4个H
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [
            # 黄烷醇特征: 逆Diels-Alder产生特征碎片
            ("RDA-A+", "A", "C6H6O3", 139.0390),  # 含3-OH的A环碎片
        ],
        "neutral_losses": ["H2O", "CO", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 4. Chalcone (查尔酮) - 如 Butein
    #    开链结构, A/B环通过 C=O-CH=CH 连接
    # ----------------------------------------------------------------
    "Chalcone": {
        "prefix": "Cha",
        "c_ring_code": "00",
        "ring_type": "open_chain",     # 无C环
        "hydrogen_delta": 0,
        "rda_cleavages": [],           # 无RDA (开链)
        "special_fragments": [
            # A环酰基正离子: C7H5O2+ (含C=O)
            ("A-ring acylium+", "A", "C7H5O2", 121.0284),
            # B环苯乙烯基正离子
            ("B-ring styryl+", "B", "C8H7O", 119.0491),
            # [M+H-B ring]+: 去B环碎片
            ("[M+H-B ring]+", "AC", "C9H7O2", 147.0441),
            # 完整 B 环正离子
            ("[B ring]+", "B", "C6H5", 77.0386),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 5. Dihydrochalcone (二氢查尔酮) - 如 Phloretin
    #    开链饱和, A/B环通过 C=O-CH2-CH2 连接
    # ----------------------------------------------------------------
    "Dihydrochalcone": {
        "prefix": "Dhc",
        "c_ring_code": "00",
        "ring_type": "open_chain",
        "hydrogen_delta": 2,           # 饱和多2H
        "rda_cleavages": [],
        "special_fragments": [
            ("A-ring acylium+", "A", "C7H5O2", 121.0284),
            ("B-ring ethyl+", "B", "C8H9O", 121.0648),
            ("[M+H-B ring]+", "AC", "C9H9O2", 149.0597),
            ("[B ring]+", "B", "C6H5", 77.0386),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 6. Isoflavanone (异黄烷酮) - 如 2'-hydroxygenistein
    #    B环在C3位, C环2,3位饱和, 有C4=O
    # ----------------------------------------------------------------
    "Isoflavanone": {
        "prefix": "Ifn",
        "c_ring_code": "00",
        "ring_type": "carbonyl",
        "hydrogen_delta": 2,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [
            # 异黄酮/异黄烷酮特征: B环直接断裂
            ("[B ring]+", "B", "C6H5", 77.0386),
            ("[M+H-B ring]+", "AC", "C9H7O2", 147.0441),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 7. Isoflavan (异黄烷) - 如 Isomucronulatol
    #    B环在C3位, C环完全饱和, 无C=O
    # ----------------------------------------------------------------
    "Isoflavan": {
        "prefix": "Ifl",
        "c_ring_code": "00",
        "ring_type": "saturated",
        "hydrogen_delta": 4,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
        ],
        "special_fragments": [
            ("[B ring]+", "B", "C6H5", 77.0386),
            ("[M+H-B ring]+", "AC", "C9H9O", 133.0648),
        ],
        "neutral_losses": ["H2O", "CH3"],
        "max_co_losses": 1,
    },

    # ----------------------------------------------------------------
    # 8. Aurone (橙酮) - 如 Sulfuretin
    #    五元环 (苯并呋喃酮), C环为 C=O-CH=C
    # ----------------------------------------------------------------
    "Aurone": {
        "prefix": "Aur",
        "c_ring_code": "00",
        "ring_type": "carbonyl",       # 五元环含C=O
        "hydrogen_delta": 0,
        "rda_cleavages": [
            ("1,3A+", "A"),            # A环保留
            ("0,2B+", "B"),            # B环保留
            ("1,3B+", "B"),
            ("0,4B+", "B"),
        ],
        "special_fragments": [
            # 橙酮特征: 苯并呋喃酮碎片
            ("Benzofuranone+", "A", "C8H5O2", 133.0284),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 9. Anthocyanidin (花青素) - 如 Cyanidin
    #    C环正离子 (oxonium), 无C4=O, 有C环芳化
    # ----------------------------------------------------------------
    "Anthocyanidin": {
        "prefix": "Ant",
        "c_ring_code": "00",
        "ring_type": "oxonium",        # C环氧鎓离子
        "hydrogen_delta": -1,          # 少1个H (正离子芳化)
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },

    # ----------------------------------------------------------------
    # 10. Flavan (黄烷) - 无C=O, 无3-OH, C环饱和
    # ----------------------------------------------------------------
    "Flavan": {
        "prefix": "Fva",
        "c_ring_code": "00",
        "ring_type": "saturated",
        "hydrogen_delta": 4,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [],
        "neutral_losses": ["H2O", "CH3"],
        "max_co_losses": 1,
    },

    # ----------------------------------------------------------------
    # 11. Pterocarpan (紫檀烷) - 如 Methylnissolin
    #     四环结构 (苯并呋喃 + 色满), 特征碎片
    # ----------------------------------------------------------------
    "Pterocarpan": {
        "prefix": "Pte",
        "c_ring_code": "00",
        "ring_type": "saturated",
        "hydrogen_delta": 4,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
        ],
        "special_fragments": [
            # 紫檀烷特征: 苯并呋喃碎片
            ("Benzofuran+", "A", "C8H7O", 119.0491),
            # 色满碎片
            ("Chroman+", "B", "C9H9O", 133.0648),
        ],
        "neutral_losses": ["H2O", "CH3", "CO"],
        "max_co_losses": 1,
    },

    # ----------------------------------------------------------------
    # 12. Xanthone (呫吨酮) - 如 Mangiferin
    #     三环骈合, C环含C=O, 无B环连接
    # ----------------------------------------------------------------
    "Xanthone": {
        "prefix": "Xan",
        "c_ring_code": "00",
        "ring_type": "carbonyl",
        "hydrogen_delta": 0,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [
            # 呫吨酮特征碎片
            ("Xanthylium+", "AC", "C13H9O", 181.0648),
            ("Benzoyl+", "A", "C7H5O", 105.0335),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },
}


# ====================================================================
# 原版 4 类的规则 (用于生成兼容的 CFIL)
# ====================================================================
ORIGINAL_RULES = {
    "Flavones": {
        "prefix": "",                 # 无前缀
        "c_ring_code": "00",
        "ring_type": "carbonyl",
        "hydrogen_delta": 0,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },
    "IsoFlavones": {
        "prefix": "Iso-",
        "c_ring_code": "00",
        "ring_type": "carbonyl",
        "hydrogen_delta": 0,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
            ("0,3A+", "A"),
            ("0,3B+", "B"),
            ("1,4B+", "B"),
        ],
        "special_fragments": [
            ("[B ring]+", "B", "C6H5", 77.0386),
            ("[M+H-B ring]+", "AC", "C9H5O2", 145.0284),
        ],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },
    "Flavonols (C-3OH)": {
        "prefix": "",
        "c_ring_code": "10",
        "ring_type": "carbonyl",
        "hydrogen_delta": 0,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2A+", "A"),
            ("0,3A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("1,2B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },
    "Flavonols (C-3OCH3)": {
        "prefix": "",
        "c_ring_code": "01",
        "ring_type": "carbonyl",
        "hydrogen_delta": 0,
        "rda_cleavages": [
            ("1,3A+", "A"),
            ("1,4A+", "A"),
            ("0,2A+", "A"),
            ("0,3A+", "A"),
            ("0,2B+", "B"),
            ("1,3B+", "B"),
            ("0,4B+", "B"),
            ("0,4A+", "A"),
        ],
        "special_fragments": [],
        "neutral_losses": ["CO", "H2O", "CH3"],
        "max_co_losses": 2,
    },
}

# 所有规则合并 (原版 + 新增)
ALL_RULES = {**ORIGINAL_RULES, **SKELETON_RULES}


def get_sm_prefix(skeleton_class):
    """获取骨架类型的 SM 命名前缀"""
    rules = ALL_RULES.get(skeleton_class, {})
    return rules.get("prefix", "")


def get_c_ring_code(skeleton_class):
    """获取骨架类型的 C 环编码"""
    rules = ALL_RULES.get(skeleton_class, {})
    return rules.get("c_ring_code", "00")
