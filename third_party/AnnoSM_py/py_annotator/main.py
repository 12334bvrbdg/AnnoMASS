#!/usr/bin/env python3
"""
AnnoSM Python 版 — 命令行入口
=================================

用法:
    # 完整流程: 筛选 + SM 注释
    python -m py_annotator.main run --mgf "Test data_mgf files/148 analytical standards/merged 73 aglycones_30 V.mgf" \\
        --tpil Library/TPIL.xlsx --cfil Library/CFIL.xlsx -o results/sm_result.xlsx

    # 仅 SM 注释（使用已有的筛选结果）
    python -m py_annotator.main sm-only --mgf <file.mgf> --cfil Library/CFIL.xlsx \\
        --screened screened_result.csv -o sm_result.xlsx

    # 仅化合物筛选
    python -m py_annotator.main screen --mgf <file.mgf> --tpil Library/TPIL.xlsx -o screened.xlsx
"""

import argparse
import sys
import time
from pathlib import Path

# 将项目根目录加入 path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from py_annotator.mgf_parser import read_mgf, spectrum_to_dataframe
from py_annotator.screener import CompoundScreener
from py_annotator.sm_annotator import SMAnnotator


def build_parser():
    parser = argparse.ArgumentParser(
        description="AnnoSM — 取代基模式 (SM) 自动注释工具 (Python 版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程
  python -m py_annotator.main run --mgf "Test data_mgf files/..." \\
      --tpil Library/TPIL.xlsx --cfil Library/CFIL.xlsx -o result.xlsx

  # 仅 SM 注释
  python -m py_annotator.main sm-only --mgf test.mgf \\
      --cfil Library/CFIL.xlsx --screened screened.csv -o sm_result.xlsx

  # 仅化合物筛选
  python -m py_annotator.main screen --mgf test.mgf \\
      --tpil Library/TPIL.xlsx -o screened.xlsx
        """,
    )
    parser.add_argument(
        "--version", action="version", version="AnnoSM Python 1.0.0"
    )

    sub = parser.add_subparsers(dest="command", help="子命令")

    # ----- run: 完整流程 -----
    p_run = sub.add_parser("run", help="完整流程: 筛选 + SM 注释")
    p_run.add_argument("--mgf", required=True, help="输入 MGF 文件路径")
    p_run.add_argument("--tpil", required=True, help="TPIL 文件路径 (.xlsx)")
    p_run.add_argument("--cfil", required=True, help="CFIL 文件路径 (.xlsx)")
    p_run.add_argument("-o", "--output", default="sm_result.xlsx", help="输出文件路径")
    p_run.add_argument(
        "--precursor-ppm", type=float, default=10, help="前体离子 ppm 容差 (默认 10)"
    )
    p_run.add_argument(
        "--fragment-ppm", type=float, default=50, help="碎片离子 ppm 容差 (默认 50)"
    )

    # ----- sm-only: 仅 SM 注释 -----
    p_sm = sub.add_parser("sm-only", help="仅 SM 注释（使用已有的筛选结果）")
    p_sm.add_argument("--mgf", required=True, help="输入 MGF 文件路径")
    p_sm.add_argument("--cfil", required=True, help="CFIL 文件路径 (.xlsx)")
    p_sm.add_argument(
        "--screened", required=True, help="已筛选的化合物结果文件 (.csv/.xlsx)"
    )
    p_sm.add_argument("-o", "--output", default="sm_result.xlsx", help="输出文件路径")
    p_sm.add_argument(
        "--fragment-ppm", type=float, default=50, help="碎片离子 ppm 容差 (默认 50)"
    )

    # ----- screen: 仅筛选 -----
    p_sc = sub.add_parser("screen", help="仅化合物筛选（基于 TPIL）")
    p_sc.add_argument("--mgf", required=True, help="输入 MGF 文件路径")
    p_sc.add_argument("--tpil", required=True, help="TPIL 文件路径 (.xlsx)")
    p_sc.add_argument("-o", "--output", default="screened_result.xlsx", help="输出文件路径")
    p_sc.add_argument(
        "--precursor-ppm", type=float, default=10, help="前体离子 ppm 容差 (默认 10)"
    )

    # ----- info: 查看 MGF 文件信息 -----
    p_info = sub.add_parser("info", help="查看 MGF 文件基本信息")
    p_info.add_argument("--mgf", required=True, help="输入 MGF 文件路径")
    p_info.add_argument("--num", type=int, default=5, help="显示前 N 个谱图信息 (默认 5)")

    return parser


def cmd_info(args):
    """查看 MGF 文件信息"""
    spectra = read_mgf(args.mgf)
    print(f"文件: {args.mgf}")
    print(f"谱图总数: {len(spectra)}")
    print()
    for i, spec in enumerate(spectra[: args.num]):
        print(f"  [{i}] {spec['title'] or f'X{i+1}'}")
        print(f"      前体离子: {spec['precursor_mz']:.4f}  m/z")
        print(f"      电荷: {spec['charge']}")
        print(f"      碎片数: {len(spec['peaks'])}")
        print()


def cmd_screen(args):
    """仅化合物筛选"""
    print(f"[1/3] 读取 MGF: {args.mgf}")
    spectra = read_mgf(args.mgf)
    print(f"      共 {len(spectra)} 个谱图")

    print(f"[2/3] 加载 TPIL: {args.tpil}")
    screener = CompoundScreener(args.tpil)

    print(f"[3/3] 筛选 (ppm={args.precursor_ppm})...")
    t0 = time.time()
    result = screener.screen_mgf(spectra, precursor_ppm=args.precursor_ppm)
    elapsed = time.time() - t0

    print(f"      完成! 耗时: {elapsed:.1f}s")
    print(f"      匹配到 {len(result)} 条记录")

    # 保存
    result.to_excel(args.output, index=False)
    print(f"      → 已保存: {args.output}")


def cmd_sm_only(args):
    """仅 SM 注释"""
    print(f"[1/4] 读取 MGF: {args.mgf}")
    spectra = read_mgf(args.mgf)
    print(f"      共 {len(spectra)} 个谱图")

    print(f"[2/4] 加载 CFIL: {args.cfil}")
    annotator = SMAnnotator(args.cfil)

    print(f"[3/4] 加载筛选结果: {args.screened}")
    ext = Path(args.screened).suffix.lower()
    if ext in (".xls", ".xlsx"):
        screened_df = pd.read_excel(args.screened)
    else:
        import pandas as pd
        screened_df = pd.read_csv(args.screened)

    print(f"      共 {screened_df['Compound ID'].nunique()} 个化合物")

    print(f"[4/4] SM 注释 (frag_ppm={args.fragment_ppm})...")
    t0 = time.time()
    result = annotator.run_from_spectra(
        spectra, screened_df=screened_df, fragment_ppm=args.fragment_ppm
    )
    elapsed = time.time() - t0

    print(f"      完成! 耗时: {elapsed:.1f}s")
    if len(result) > 0:
        print(f"      共 {result['Compound ID'].nunique()} 个化合物被注释")
        print(f"      共 {len(result)} 条 SM 记录")
        print()
        # 显示前几行
        display_cols = [c for c in ["Compound ID", "Substitution modes", "Score", "Class"] if c in result.columns]
        print(result[display_cols].head(10).to_string(index=False))
    else:
        print("      (无结果)")

    # 保存
    result.to_excel(args.output, index=False)
    print(f"      → 已保存: {args.output}")


def cmd_run(args):
    """完整流程: 筛选 + SM 注释"""
    import pandas as pd

    print(f"[1/5] 读取 MGF: {args.mgf}")
    spectra = read_mgf(args.mgf)
    print(f"      共 {len(spectra)} 个谱图")

    print(f"[2/5] 加载 TPIL: {args.tpil}")
    screener = CompoundScreener(args.tpil)

    print(f"[3/5] 筛选化合物 (precursor_ppm={args.precursor_ppm})...")
    t0 = time.time()
    screened = screener.screen_mgf(spectra, precursor_ppm=args.precursor_ppm)
    t1 = time.time()
    print(f"      耗时: {t1 - t0:.1f}s, 匹配到 {len(screened)} 条记录")

    print(f"[4/5] 加载 CFIL: {args.cfil}")
    annotator = SMAnnotator(args.cfil)

    print(f"[5/5] SM 注释 (fragment_ppm={args.fragment_ppm})...")
    t2 = time.time()
    result = annotator.run_from_spectra(
        spectra, screened_df=screened, fragment_ppm=args.fragment_ppm
    )
    t3 = time.time()
    print(f"      耗时: {t3 - t2:.1f}s")

    if len(result) > 0:
        num_compounds = result["Compound ID"].nunique()
        print(f"      共 {num_compounds} 个化合物, {len(result)} 条 SM 记录")
        display_cols = [c for c in ["Compound ID", "Substitution modes", "Score", "Class"] if c in result.columns]
        print()
        print(result[display_cols].head(15).to_string(index=False))
    else:
        print("      (无结果)")

    # 同时保存筛选结果
    output_path = Path(args.output)
    screened_path = output_path.parent / f"screened_{output_path.name}"

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="SM Results", index=False)
        screened.to_excel(writer, sheet_name="Screened Compounds", index=False)

    print(f"      → 已保存: {args.output}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "screen":
        cmd_screen(args)
    elif args.command == "sm-only":
        cmd_sm_only(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
