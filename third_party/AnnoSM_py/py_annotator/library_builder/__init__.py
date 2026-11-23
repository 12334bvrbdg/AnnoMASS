# library_builder - 扩展版 TPIL/CFIL 库构建工具
"""
从 DeepMASS2 黄酮结构数据库构建扩展版 TPIL/CFIL 库。

新增 12 种黄酮骨架类型:
  Flavanone, Flavanonol, Flavan3ol, Chalcone, Dihydrochalcone,
  Isoflavanone, Isoflavan, Aurone, Anthocyanidin, Flavan,
  Pterocarpan, Xanthone

加上原版的 4 类:
  Flavones, IsoFlavones, Flavonols (C-3OH), Flavonols (C-3OCH3)

共 16 种黄酮骨架类型。
"""

from .fragmentation_rules import SKELETON_RULES, ORIGINAL_CLASSES
from .tpil_builder import TPILBuilder
from .cfil_builder import CFILBuilder
