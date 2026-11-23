#!/usr/bin/env python3
"""
一键构建扩展版 TPIL/CFIL 库
==============================

用法:
    python -m py_annotator.library_builder.build_all \\
        --structure-db "/path/to/Annomass/Stru/FlavonoidStructureDB-optimized.csv" \\
        --flavclass-path "/path/to/Annomass/core" \\
        --output-dir Library/

    # 仅构建 CFIL (不需要结构数据库)
    python -m py_annotator.library_builder.build_all --cfil-only --output-dir Library/

    # 仅构建 TPIL
    python -m py_annotator.library_builder.build_all --tpil-only \\
        --structure-db "..." --flavclass-path "..." --output-dir Library/
"""

import argparse
import sys
import time
from pathlib import Path

# 将项目根目录加入 path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def build_tpil(args):
    """构建 TPIL"""
    from .tpil_builder import TPILBuilder

    print("=" * 70)
    print("  构建 TPIL (前体离子库)")
    print("=" * 70)

    builder = TPILBuilder(
        structure_db_path=args.structure_db,
        flavclass_module_path=args.flavclass_path,
    )

    t0 = time.time()
    tpil_df = builder.build(aglycone_only=True, max_compounds=args.max_compounds)
    t1 = time.time()
    print(f"\nTPIL 构建耗时: {t1 - t0:.1f}s")

    # 生成去重汇总
    summary = builder.build_summary(tpil_df)
    print(f"去重后: {len(summary)} 条 (骨架×OH×OCH3 组合)")

    # 与原版合并
    combined = builder.merge_with_original(summary, args.original_tpil)
    print(f"合并后总计: {len(combined)} 条")

    # 保存
    output_path = Path(args.output_dir) / "TPIL_extended.xlsx"
    builder.save(combined, output_path)
    return combined


def build_cfil(args):
    """构建 CFIL"""
    from .cfil_builder import CFILBuilder

    print("=" * 70)
    print("  构建 CFIL (特征碎片离子库)")
    print("=" * 70)

    builder = CFILBuilder(
        max_oh_per_ring=args.max_oh,
        max_och3_per_ring=args.max_och3,
    )

    t0 = time.time()
    cfil_df = builder.build()
    t1 = time.time()
    print(f"\nCFIL 构建耗时: {t1 - t0:.1f}s")

    # 与原版合并
    combined = builder.merge_with_original(cfil_df, args.original_cfil)
    print(f"合并后总计: {len(combined)} 条")

    # 保存
    output_path = Path(args.output_dir) / "CFIL_extended.xlsx"
    builder.save(combined, output_path)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="构建扩展版 AnnoSM TPIL/CFIL 库 (16 种黄酮骨架)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--structure-db",
        default="Stru/FlavonoidStructureDB-dedup.csv",
        help="DeepMASS2 结构数据库路径",
    )
    parser.add_argument(
        "--flavclass-path",
        default="core",
        help="DeepMASS2 core 目录路径 (含 flavclass_filter.py)",
    )
    parser.add_argument(
        "--output-dir",
        default="Library",
        help="输出目录 (默认 Library/)",
    )
    parser.add_argument(
        "--original-tpil",
        default=None,
        help="原版 TPIL.xlsx 路径 (默认 Library/TPIL.xlsx)",
    )
    parser.add_argument(
        "--original-cfil",
        default=None,
        help="原版 CFIL.xlsx 路径 (默认 Library/CFIL.xlsx)",
    )
    parser.add_argument(
        "--tpil-only", action="store_true", help="仅构建 TPIL"
    )
    parser.add_argument(
        "--cfil-only", action="store_true", help="仅构建 CFIL"
    )
    parser.add_argument(
        "--max-compounds", type=int, default=50000,
        help="最大处理化合物数 (默认 50000)",
    )
    parser.add_argument(
        "--max-oh", type=int, default=5,
        help="每环最大 OH 数 (默认 5)",
    )
    parser.add_argument(
        "--max-och3", type=int, default=5,
        help="每环最大 OCH3 数 (默认 5)",
    )

    args = parser.parse_args()

    # 设置默认路径
    if args.original_tpil is None:
        args.original_tpil = Path(args.output_dir) / "TPIL.xlsx"
    if args.original_cfil is None:
        args.original_cfil = Path(args.output_dir) / "CFIL.xlsx"

    # 构建
    if args.tpil_only:
        build_tpil(args)
    elif args.cfil_only:
        build_cfil(args)
    else:
        build_tpil(args)
        print()
        build_cfil(args)

    print("\n" + "=" * 70)
    print("  构建完成!")
    print("=" * 70)
    print(f"\n扩展版库已保存到: {args.output_dir}/")
    print(f"  - TPIL_extended.xlsx")
    print(f"  - CFIL_extended.xlsx")
    print(f"\n使用方法:")
    print(f"  python -m py_annotator.main run \\")
    print(f"      --mgf your_query.mgf \\")
    print(f"      --tpil {args.output_dir}/TPIL_extended.xlsx \\")
    print(f"      --cfil {args.output_dir}/CFIL_extended.xlsx \\")
    print(f"      -o results/extended_result.xlsx")


if __name__ == "__main__":
    main()
