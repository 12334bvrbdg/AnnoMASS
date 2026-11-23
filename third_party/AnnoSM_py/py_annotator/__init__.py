# py_annotator — AnnoSM 算法的 Python 实现
# 取代基模式 (Substituent Mode, SM) 注释工具
"""
AnnoSM Python 实现
===================
基于特征碎片离子库 (CFIL) 和前体离子库 (TPIL) 对黄酮类化合物的
取代基模式进行自动化注释。

核心逻辑移植自 AnnoSM.R (R Shiny 应用)。

模块:
    mgf_parser   — MGF 谱图文件解析
    screener     — TPIL 前体离子筛选
    sm_annotator — CFIL 特征碎片匹配 + SM 聚合 + 分类 + 位点注释
    main         — 命令行入口

用法:
    from py_annotator import SMAnnotator
    annotator = SMAnnotator("Library/TPIL.xlsx", "Library/CFIL.xlsx")
    result = annotator.run("path/to/file.mgf", precursor_ppm=10, fragment_ppm=50)
"""

__version__ = "1.0.0"

from .sm_annotator import SMAnnotator
from .mgf_parser import read_mgf
from .screener import CompoundScreener
