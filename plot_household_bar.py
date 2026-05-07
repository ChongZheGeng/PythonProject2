import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


UNIT_LABELS = {
    "wanhu": "万户",
    "raw": "户",
}


INDICATOR_LABELS = {
    "family_households": "家庭户数",
    "total_households": "总户数",
    "collective_households": "集体户数",
}


def check_excel_dependency() -> None:
    """检查 openpyxl 依赖是否可用。"""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("错误：当前 Python 虚拟环境缺少 openpyxl，无法读取 .xlsx 文件。")
        print("请在 PyCharm 终端运行：")
        print(r"D:\PythonProject2\.venv\Scripts\python.exe -m pip install openpyxl")
        sys.exit(1)


def resolve_file_path(file_path: str) -> Path:
    """解析 Excel 文件路径，优先当前工作目录，其次脚本目录。"""
    candidate = Path(file_path)
    if candidate.is_file():
        return candidate

    script_dir_candidate = Path(__file__).resolve().parent / file_path
    if script_dir_candidate.is_file():
        return script_dir_candidate

    print(f"错误：未找到 Excel 文件：{file_path}")
    print(f"已尝试路径：{candidate.resolve()} 和 {script_dir_candidate.resolve()}")
    sys.exit(1)


def load_household_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """加载指定工作表数据。"""
    resolved = resolve_file_path(file_path)
    try:
        df = pd.read_excel(resolved, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as exc:
        print(f"错误：无法读取工作表“{sheet_name}”：{exc}")
        sys.exit(1)
    return df


def filter_household_data(df: pd.DataFrame, region: str, indicator: str) -> pd.DataFrame:
    """按地区和指标过滤数据。"""
    if indicator not in df.columns:
        print(f"错误：指标“{indicator}”不存在。")
        print("当前表字段名如下：")
        print(", ".join(df.columns.astype(str)))
        sys.exit(1)

    required_cols = {"region_short", "census_year", indicator, "family_households"}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误：缺少必要字段：{', '.join(missing_cols)}")
        sys.exit(1)

    filtered = df[df["region_short"].astype(str) == str(region)].copy()
    if filtered.empty:
        available_regions = sorted(df["region_short"].dropna().astype(str).unique().tolist())
        print(f"错误：未找到地区“{region}”的数据。")
        print("当前可用地区列表：")
        print("、".join(available_regions))
        sys.exit(1)

    filtered["family_households"] = pd.to_numeric(filtered["family_households"], errors="coerce")
    filtered[indicator] = pd.to_numeric(filtered[indicator], errors="coerce")
    filtered["census_year"] = pd.to_numeric(filtered["census_year"], errors="coerce")

    filtered = filtered[filtered["family_households"].notna()].copy()
    filtered = filtered[filtered["census_year"].notna()].copy()
    filtered = filtered.sort_values("census_year", ascending=True)

    if filtered.empty:
        print(f"错误：地区“{region}”在 family_households 非空条件下无可用数据。")
        sys.exit(1)

    return filtered


def prepare_plot_data(df: pd.DataFrame, indicator: str, unit: str) -> pd.DataFrame:
    """准备绘图字段与单位换算。"""
    if unit not in UNIT_LABELS:
        print(f"错误：不支持的单位“{unit}”，仅支持 raw 或 wanhu。")
        sys.exit(1)

    data = df[["census_year", indicator]].copy()
    data[indicator] = pd.to_numeric(data[indicator], errors="coerce")
    data = data.dropna(subset=[indicator])

    if unit == "wanhu":
        data["plot_value"] = data[indicator] / 10000.0
    else:
        data["plot_value"] = data[indicator]

    data = data.dropna(subset=["plot_value"]).copy()
    data["census_year"] = data["census_year"].astype(int)

    if data.empty:
        print("错误：处理后没有可绘制的数据。")
        sys.exit(1)

    return data


def plot_household_bar(df: pd.DataFrame, region: str, indicator: str, unit: str, output_path: str) -> None:
    """绘制家庭户数柱状图。"""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in available_fonts:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    indicator_label = INDICATOR_LABELS.get(indicator, indicator)
    unit_label = UNIT_LABELS[unit]

    years = df["census_year"].tolist()
    values = df["plot_value"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bars = ax.bar(
        years,
        values,
        color="#4C78A8",
        edgecolor="#2F4B6C",
        linewidth=1.0,
        width=6,
    )

    if region == "全国" and indicator == "family_households":
        title = "历次人口普查全国家庭户数变化"
    else:
        title = f"历次人口普查{region}{indicator_label}变化"

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("普查年份", fontsize=12)
    ax.set_ylabel(f"{indicator_label}（{unit_label}）", fontsize=12)

    ax.set_xticks(years)
    ax.set_xticklabels([f"{year}年" for year in years], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(axis="y", linestyle="--", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = max(values)
    upper = max_val * 1.15 if max_val > 0 else 1
    ax.set_ylim(0, upper)

    offset = max_val * 0.02 if max_val > 0 else 0.1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存家庭户数柱状图：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制历次人口普查家庭户数柱状图")
    parser.add_argument("--file", default="人口普查数据_格式修正版.xlsx", help="Excel 文件路径")
    parser.add_argument("--sheet", default="可视化主表", help="工作表名称")
    parser.add_argument("--region", default="全国", help="地区简称（region_short）")
    parser.add_argument("--indicator", default="family_households", help="指标字段名")
    parser.add_argument("--unit", default="wanhu", choices=["wanhu", "raw"], help="单位：wanhu 或 raw")
    parser.add_argument("--output", default="outputs/household_bar_全国_family_households.png", help="输出图片路径")

    args = parser.parse_args()

    unit_text = "万户" if args.unit == "wanhu" else "户"
    print("当前绘图参数：")
    print(f"文件：{args.file}")
    print(f"工作表：{args.sheet}")
    print(f"地区：{args.region}")
    print(f"指标：{args.indicator}")
    print(f"单位：{unit_text}")
    print(f"输出路径：{args.output}")

    check_excel_dependency()
    df = load_household_data(args.file, args.sheet)
    filtered = filter_household_data(df, args.region, args.indicator)
    plot_df = prepare_plot_data(filtered, args.indicator, args.unit)
    plot_household_bar(plot_df, args.region, args.indicator, args.unit, args.output)


if __name__ == "__main__":
    main()
