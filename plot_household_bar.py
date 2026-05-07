import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def beautify_3d_axes(ax) -> None:
    """统一 3D 坐标轴轻量化样式。"""
    # 透明 pane，减少默认灰色厚重感
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.45))

    # 弱化坐标轴线
    try:
        ax.xaxis.line.set_color((0.75, 0.75, 0.75, 0.6))
        ax.yaxis.line.set_color((0.82, 0.82, 0.82, 0.25))
        ax.zaxis.line.set_color((0.75, 0.75, 0.75, 0.65))
    except Exception:
        pass


def plot_household_bar_3d(df: pd.DataFrame, region: str, indicator: str, unit: str, output_path: str, style: str = "clean3d") -> None:
    """绘制家庭户数柱状图（clean3d 或 pseudo3d）。"""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in available_fonts:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    indicator_label = INDICATOR_LABELS.get(indicator, indicator)
    unit_label = UNIT_LABELS[unit]

    years = df["census_year"].tolist()
    values = np.array(df["plot_value"].tolist(), dtype=float)

    fig = plt.figure(figsize=(10, 6.8), dpi=300)
    fig.patch.set_facecolor("white")

    if style == "pseudo3d":
        ax = fig.add_subplot(111)
        x = np.arange(len(years), dtype=float)
        width = 0.56
        colors = ["#8FB6FF", "#6F9EF8", "#4F81E8"]

        bars = ax.bar(x, values, width=width, color=colors[: len(x)], edgecolor="#3D5A80", linewidth=0.9, alpha=0.95, zorder=3)

        # 用偏移多边形制造轻 2.5D 顶面与右侧面
        depth_x, depth_y = 0.08, values.max() * 0.018 if values.max() > 0 else 0.1
        for rect, c in zip(bars, colors):
            x0, w, h = rect.get_x(), rect.get_width(), rect.get_height()
            top = plt.Polygon([[x0, h], [x0 + w, h], [x0 + w + depth_x, h + depth_y], [x0 + depth_x, h + depth_y]],
                              closed=True, facecolor="#9FC1FF", edgecolor="#3D5A80", linewidth=0.6, alpha=0.9, zorder=2)
            side = plt.Polygon([[x0 + w, 0], [x0 + w, h], [x0 + w + depth_x, h + depth_y], [x0 + w + depth_x, depth_y]],
                               closed=True, facecolor="#5B8FF9", edgecolor="#3D5A80", linewidth=0.6, alpha=0.9, zorder=1)
            ax.add_patch(top)
            ax.add_patch(side)

        if region == "全国" and indicator == "family_households":
            title = "历次人口普查全国家庭户数变化"
        else:
            title = f"历次人口普查{region}{indicator_label}变化"

        ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
        ax.set_xlabel("普查年份", fontsize=13, labelpad=10)
        ax.set_ylabel(f"{indicator_label}（{unit_label}）", fontsize=13, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{y}年" for y in years], fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, values.max() * 1.15 if values.max() > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#D9D9D9", alpha=0.75, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BFBFBF")
        ax.spines["bottom"].set_color("#BFBFBF")

        offset = values.max() * 0.02 if values.max() > 0 else 0.1
        for i, v in enumerate(values):
            ax.text(i, v + offset, f"{v:.1f}", ha="center", va="bottom", fontsize=11, color="#303030")
    else:
        ax = fig.add_subplot(111, projection="3d")

        x = np.arange(len(years), dtype=float)
        y = np.zeros(len(years), dtype=float)
        z = np.zeros(len(years), dtype=float)
        dx = np.full(len(years), 0.42, dtype=float)
        dy = np.full(len(years), 0.22, dtype=float)
        dz = values

        colors = ["#8FB6FF", "#6F9EF8", "#4F81E8"]

        ax.bar3d(x, y, z, dx, dy, dz, color=colors[: len(x)], edgecolor="#3D5A80", linewidth=0.65, alpha=0.95, shade=True)

        if region == "全国" and indicator == "family_households":
            title = "历次人口普查全国家庭户数变化"
        else:
            title = f"历次人口普查{region}{indicator_label}变化"

        ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
        ax.set_xlabel("普查年份", fontsize=13, labelpad=10)
        ax.set_ylabel("")
        ax.set_zlabel(f"{indicator_label}（{unit_label}）", fontsize=13, labelpad=12)

        ax.set_xticks(x + dx / 2)
        ax.set_xticklabels([f"{year}年" for year in years], fontsize=12, rotation=0, ha="center")
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="z", labelsize=11)

        # 仅保留很浅的 z 向网格
        ax.xaxis._axinfo["grid"].update(color=(1, 1, 1, 0), linewidth=0)
        ax.yaxis._axinfo["grid"].update(color=(1, 1, 1, 0), linewidth=0)
        ax.zaxis._axinfo["grid"].update(color=(0.85, 0.85, 0.85, 0.75), linestyle="--", linewidth=0.6)

        beautify_3d_axes(ax)
        ax.view_init(elev=18, azim=-58)
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")

        max_val = float(values.max()) if len(values) else 0.0
        upper = max_val * 1.15 if max_val > 0 else 1.0
        ax.set_zlim(0, upper)

        for i, val in enumerate(values):
            z_text = val * 1.02 if val > 0 else upper * 0.02
            ax.text(x[i] + dx[i] / 2, y[i] + dy[i] / 2, z_text, f"{val:.1f}", ha="center", va="bottom", fontsize=11, color="#2F2F2F")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.10)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"已保存家庭户数柱状图：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制历次人口普查家庭户数柱状图")
    parser.add_argument("--file", default="人口普查数据_格式修正版.xlsx", help="Excel 文件路径")
    parser.add_argument("--sheet", default="可视化主表", help="工作表名称")
    parser.add_argument("--region", default="全国", help="地区简称（region_short）")
    parser.add_argument("--indicator", default="family_households", help="指标字段名")
    parser.add_argument("--unit", default="wanhu", choices=["wanhu", "raw"], help="单位：wanhu 或 raw")
    parser.add_argument("--output", default="outputs/household_bar_3d_全国_family_households.png", help="输出图片路径")
    parser.add_argument("--style", default="clean3d", choices=["clean3d", "pseudo3d"], help="绘图风格：clean3d（轻3D）或 pseudo3d（2.5D）")

    args = parser.parse_args()

    unit_text = "万户" if args.unit == "wanhu" else "户"
    print("当前绘图参数：")
    print(f"文件：{args.file}")
    print(f"工作表：{args.sheet}")
    print(f"地区：{args.region}")
    print(f"指标：{args.indicator}")
    print(f"单位：{unit_text}")
    print(f"输出路径：{args.output}")
    print(f"绘图风格：{args.style}")

    check_excel_dependency()
    df = load_household_data(args.file, args.sheet)
    filtered = filter_household_data(df, args.region, args.indicator)
    plot_df = prepare_plot_data(filtered, args.indicator, args.unit)
    plot_household_bar_3d(plot_df, args.region, args.indicator, args.unit, args.output, args.style)


if __name__ == "__main__":
    main()
