"""
绘制图3-1 省份聚类类型空间分布图（Geopandas + Matplotlib 静态版）

依赖安装（按需）：
    pip install pandas openpyxl geopandas matplotlib
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


# =========================
# 基本配置
# =========================
INPUT_FILENAME = "省份聚类结果_K4.xlsx"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = BASE_DIR / "图3-1_省份聚类类型空间分布图.png"

TITLE = "图3-1 省份聚类类型空间分布图"

CATEGORY_LABELS: Dict[int, str] = {
    1: "类别1：都市型",
    2: "类别2：高活力/转型型",
    3: "类别3：东北收缩型",
    4: "类别4：主体混合型",
}

# 论文风格：低饱和、清晰
CATEGORY_COLORS: Dict[int, str] = {
    1: "#4E79A7",
    2: "#76A86B",
    3: "#C86C6B",
    4: "#9A8B7A",
}

DEFAULT_FILL = "#E6E6E6"  # 未赋类省份（如港澳台缺失时）
EDGE_COLOR = "#666666"


# =========================
# 路径查找
# =========================
def find_input_excel() -> Tuple[Path, List[Path]]:
    """按优先顺序查找输入 Excel 文件。"""
    candidates = [
        BASE_DIR / INPUT_FILENAME,
        Path.cwd() / INPUT_FILENAME,
        Path("D:/PythonProject2/省份聚类结果_K4.xlsx"),
        Path("D:/PythonProject2/data/省份聚类结果_K4.xlsx"),
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p, candidates

    msg_lines = [
        f"未找到输入文件：{INPUT_FILENAME}",
        f"当前工作目录：{Path.cwd()}",
        f"脚本所在目录：{BASE_DIR}",
        "已尝试以下路径：",
    ]
    for idx, p in enumerate(candidates, start=1):
        msg_lines.append(f"{idx}. {p}")
    msg_lines.append("请将 Excel 文件放到脚本同级目录，或手动修改 INPUT_FILENAME。")
    raise FileNotFoundError("\n".join(msg_lines))


def find_china_boundary_file() -> Tuple[Path, List[Path]]:
    """按要求优先顺序查找中国省级边界文件。"""
    candidates = [
        Path("D:/PythonProject2/china_provinces.geojson"),
        Path("D:/PythonProject2/data/china_provinces.geojson"),
        Path("D:/PythonProject2/china_provinces.shp"),
        Path("D:/PythonProject2/data/china_provinces.shp"),
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p, candidates

    msg_lines = [
        "未找到中国省级行政区边界文件（GeoJSON / Shapefile）。",
        f"当前工作目录：{Path.cwd()}",
        f"脚本所在目录：{BASE_DIR}",
        "已尝试以下地图文件路径：",
    ]
    for idx, p in enumerate(candidates, start=1):
        msg_lines.append(f"{idx}. {p}")
    msg_lines.append("请下载并放入中国省级行政区 GeoJSON（推荐文件名 china_provinces.geojson）后重试。")
    raise FileNotFoundError("\n".join(msg_lines))


# =========================
# 名称标准化
# =========================
def normalize_province_name(raw_name: str) -> List[str]:
    """将原始省份名称标准化为统一名称。

    返回 list 的原因：
    - 常规情况返回单一省份；
    - 特殊情况“四川（含重庆）”会拆分为“四川”“重庆”。
    """
    if pd.isna(raw_name):
        return []

    name = str(raw_name).strip()
    if not name:
        return []

    name = re.sub(r"[\s\u3000]+", "", name)

    if re.search(r"四川.*含.*重庆", name):
        # 由于第四次人口普查时期重庆尚未单列，为保证空间分布图完整，
        # 因此在可视化中将四川和重庆赋为同一类别。
        return ["四川", "重庆"]

    direct_map = {
        "北京市": "北京",
        "天津市": "天津",
        "上海市": "上海",
        "重庆市": "重庆",
        "河北省": "河北",
        "内蒙古自治区": "内蒙古",
        "广西壮族自治区": "广西",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
        "西藏自治区": "西藏",
    }
    if name in direct_map:
        return [direct_map[name]]

    name = (
        name.replace("省", "")
        .replace("市", "")
        .replace("特别行政区", "")
        .replace("壮族自治区", "")
        .replace("回族自治区", "")
        .replace("维吾尔自治区", "")
        .replace("自治区", "")
    )

    alias_map = {
        "内蒙古": "内蒙古",
        "广西": "广西",
        "宁夏": "宁夏",
        "新疆": "新疆",
        "西藏": "西藏",
        "香港": "香港",
        "澳门": "澳门",
        "台湾": "台湾",
        "黑龙江": "黑龙江",
    }
    return [alias_map.get(name, name)]


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """自动识别“省份列”和“类别列”。"""
    col_names = list(df.columns)
    province_keywords = ["省", "市", "地区", "行政区", "province", "prov", "区域", "省份"]
    category_keywords = ["类", "类别", "cluster", "聚类", "type", "分组"]

    def score_column(col: str, keywords: List[str]) -> int:
        c = str(col).strip().lower()
        return sum(2 for kw in keywords if kw.lower() in c)

    province_scores = {c: score_column(c, province_keywords) for c in col_names}
    category_scores = {c: score_column(c, category_keywords) for c in col_names}

    province_col = max(province_scores, key=province_scores.get)
    category_col = max(category_scores, key=category_scores.get)

    if province_scores[province_col] == 0:
        province_col = col_names[0]

    if category_scores[category_col] == 0:
        best_col = None
        best_match = -1
        for c in col_names:
            series = pd.to_numeric(df[c], errors="coerce")
            valid = series.dropna()
            if len(valid) == 0:
                continue
            match = ((valid >= 1) & (valid <= 4)).sum()
            if match > best_match:
                best_match = match
                best_col = c
        if best_col is not None:
            category_col = best_col

    return province_col, category_col


def detect_geo_province_column(gdf: gpd.GeoDataFrame) -> str:
    """自动识别边界文件中的省名字段。"""
    candidates = ["省份", "省", "NAME", "NAME_CHN", "name", "name_zh", "prov_name", "province"]
    cols = list(gdf.columns)

    for c in candidates:
        if c in cols:
            return c

    for c in cols:
        if "name" in str(c).lower() or "省" in str(c):
            return c

    raise ValueError(f"无法识别边界文件中的省份名称字段，当前字段为：{cols}")


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    input_excel, _ = find_input_excel()
    print(f"使用输入文件：{input_excel.resolve()}")

    boundary_file, _ = find_china_boundary_file()
    print(f"使用地图边界文件：{boundary_file.resolve()}")

    df = pd.read_excel(input_excel)
    if df.empty:
        raise ValueError("Excel 文件为空，无法绘图。")

    province_col, category_col = detect_columns(df)
    print(f"识别到省份列：{province_col}")
    print(f"识别到类别列：{category_col}")

    rows = []
    for _, row in df.iterrows():
        provinces = normalize_province_name(row[province_col])
        if not provinces:
            continue

        raw_category = row[category_col]
        cat = pd.to_numeric(pd.Series([raw_category]), errors="coerce").iloc[0]
        if pd.isna(cat):
            m = re.search(r"(\d+)", str(raw_category))
            if m:
                cat = int(m.group(1))

        cat = int(cat) if pd.notna(cat) else None
        if cat not in {1, 2, 3, 4}:
            print(f"[警告] 类别值异常，已跳过：省份={row[province_col]}, 类别={raw_category}")
            continue

        for p in provinces:
            rows.append((p, cat))

    if not rows:
        raise ValueError("清洗后无有效数据，请检查输入表格。")

    clean_df = pd.DataFrame(rows, columns=["省份", "类别"]).drop_duplicates(subset=["省份"], keep="last")

    gdf = gpd.read_file(boundary_file)
    name_col = detect_geo_province_column(gdf)
    gdf["省份标准化"] = gdf[name_col].apply(lambda x: normalize_province_name(x)[0] if normalize_province_name(x) else None)

    merged = gdf.merge(clean_df, left_on="省份标准化", right_on="省份", how="left")
    merged["填充色"] = merged["类别"].map(CATEGORY_COLORS).fillna(DEFAULT_FILL)

    missing_special = [p for p in ["香港", "澳门", "台湾"] if p in set(merged["省份标准化"].dropna()) and p not in set(clean_df["省份"])]
    if missing_special:
        print(f"[提示] {missing_special} 在 Excel 中无类别，已使用浅灰色显示。")

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    merged.plot(
        ax=ax,
        color=merged["填充色"],
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
    )

    ax.set_title(TITLE, fontsize=16, fontweight="bold", pad=12)
    ax.set_axis_off()

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", label=CATEGORY_LABELS[k], markerfacecolor=v, markersize=10)
        for k, v in CATEGORY_COLORS.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", label="未分类/缺失（港澳台等）", markerfacecolor=DEFAULT_FILL, markersize=10)
    )
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)

    print(f"PNG 图片已保存：{OUTPUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
