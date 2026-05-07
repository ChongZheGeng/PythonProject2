"""
绘制图3-1 省份聚类类型空间分布图

依赖安装（按需）：
    pip install pandas openpyxl pyecharts snapshot-selenium selenium

说明：
1) 本脚本读取文件名：省份聚类结果_K4.xlsx
2) 自动按多个本地路径顺序查找输入文件（优先脚本同级目录）
3) 使用 pyecharts 绘制中国省级行政区地图，并导出为高分辨率 PNG（300 dpi）
4) 若运行 snapshot-selenium 时报错，请检查本机是否已安装 Chrome/Chromium 与对应版本 chromedriver
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import CurrentConfig
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

# 使用稳定可访问的 pyecharts 静态资源地址
CurrentConfig.ONLINE_HOST = "https://assets.pyecharts.org/assets/v5/"


# =========================
# 基本配置
# =========================
INPUT_FILENAME = "省份聚类结果_K4.xlsx"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_HTML = BASE_DIR / "图3-1_省份聚类类型空间分布图.html"
OUTPUT_PNG = BASE_DIR / "图3-1_省份聚类类型空间分布图.png"

TITLE = "图3-1 省份聚类类型空间分布图"
SUBTITLE = "（按省级行政区聚类结果着色）"

# 类别映射：用于图例显示
CATEGORY_LABELS: Dict[int, str] = {
    1: "类别1：都市型",
    2: "类别2：高活力/转型型",
    3: "类别3：东北收缩型",
    4: "类别4：主体混合型",
}

# 论文风格配色（清晰但不过艳）
CATEGORY_COLORS: Dict[int, str] = {
    1: "#4E79A7",  # 蓝
    2: "#59A14F",  # 绿
    3: "#E15759",  # 红
    4: "#9C755F",  # 褐
}


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
    msg_lines.append("请将 Excel 文件放到 plot_province_cluster_map.py 同级目录，或手动修改 INPUT_EXCEL。")

    raise FileNotFoundError("\n".join(msg_lines))


# =========================
# 名称标准化
# =========================
def normalize_province_name(raw_name: str) -> List[str]:
    """将原始省份名称标准化为 pyecharts 中国地图常见名称。

    返回 list，原因：
    - 常规情况返回单一省份；
    - 特殊情况“四川（含重庆）”会拆分为“四川”“重庆”。
    """
    if pd.isna(raw_name):
        return []

    name = str(raw_name).strip()
    if not name:
        return []

    # 去除空格和常见全角空格
    name = re.sub(r"[\s\u3000]+", "", name)

    # 特殊处理：四川（含重庆）
    if re.search(r"四川.*含.*重庆", name):
        # 说明：第四次人口普查时期重庆尚未单列，
        # 为保证当前省级空间分布图完整，视觉上将该记录同时赋给四川和重庆。
        return ["四川", "重庆"]

    # 常见后缀清理
    name = (
        name.replace("省", "")
        .replace("市", "")
        .replace("特别行政区", "")
        .replace("壮族自治区", "")
        .replace("回族自治区", "")
        .replace("维吾尔自治区", "")
        .replace("自治区", "")
    )

    # 常见别名统一
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
    name = alias_map.get(name, name)

    return [name]


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """自动识别“省份列”和“类别列”。"""
    col_names = list(df.columns)

    province_keywords = ["省", "市", "地区", "行政区", "province", "prov", "区域", "省份"]
    category_keywords = ["类", "类别", "cluster", "聚类", "type", "分组"]

    def score_column(col: str, keywords: List[str]) -> int:
        c = str(col).strip().lower()
        score = 0
        for kw in keywords:
            if kw.lower() in c:
                score += 2
        return score

    # 初步按列名打分
    province_scores = {c: score_column(c, province_keywords) for c in col_names}
    category_scores = {c: score_column(c, category_keywords) for c in col_names}

    province_col = max(province_scores, key=province_scores.get)
    category_col = max(category_scores, key=category_scores.get)

    # 若打分都很弱，启发式基于内容判断
    if province_scores[province_col] == 0:
        province_col = col_names[0]

    if category_scores[category_col] == 0:
        # 优先找取值在 1~4 的列
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


def main() -> None:
    input_excel, _attempted_paths = find_input_excel()
    print(f"使用输入文件：{input_excel.resolve()}")

    df = pd.read_excel(input_excel)
    if df.empty:
        raise ValueError("Excel 文件为空，无法绘图。")

    province_col, category_col = detect_columns(df)
    print(f"识别到省份列：{province_col}")
    print(f"识别到类别列：{category_col}")

    # 清洗并展开数据
    rows = []
    for _, row in df.iterrows():
        raw_province = row[province_col]
        raw_category = row[category_col]

        provinces = normalize_province_name(raw_province)
        if not provinces:
            continue

        cat = pd.to_numeric(pd.Series([raw_category]), errors="coerce").iloc[0]
        if pd.isna(cat):
            # 尝试从字符串提取数字
            m = re.search(r"(\d+)", str(raw_category))
            if m:
                cat = int(m.group(1))
        cat = int(cat) if pd.notna(cat) else None

        if cat not in {1, 2, 3, 4}:
            print(f"[警告] 类别值异常，已跳过：省份={raw_province}, 类别={raw_category}")
            continue

        for p in provinces:
            rows.append((p, cat))

    if not rows:
        raise ValueError("清洗后无有效数据，请检查输入表格。")

    clean_df = pd.DataFrame(rows, columns=["省份", "类别"]).drop_duplicates(subset=["省份"], keep="last")

    # 全部省级行政区（pyecharts 名称体系）
    all_regions = [
        "北京", "天津", "上海", "重庆", "河北", "河南", "云南", "辽宁", "黑龙江", "湖南", "安徽", "山东",
        "新疆", "江苏", "浙江", "江西", "湖北", "广西", "甘肃", "山西", "内蒙古", "陕西", "吉林", "福建",
        "贵州", "广东", "青海", "西藏", "四川", "宁夏", "海南", "台湾", "香港", "澳门",
    ]

    clean_map = dict(zip(clean_df["省份"], clean_df["类别"]))

    # 打印匹配异常信息
    input_only = sorted(set(clean_df["省份"]) - set(all_regions))
    missing_in_input = sorted(set(all_regions) - set(clean_df["省份"]))

    if input_only:
        print("[提示] 以下省份名称未能匹配到底图区域：", input_only)
    if missing_in_input:
        print("[提示] 以下省份在数据中缺失，将显示为默认底色：", missing_in_input)

    data_pair = [(r, clean_map.get(r, None)) for r in all_regions]

    pieces = [
        {"min": 1, "max": 1, "label": CATEGORY_LABELS[1], "color": CATEGORY_COLORS[1]},
        {"min": 2, "max": 2, "label": CATEGORY_LABELS[2], "color": CATEGORY_COLORS[2]},
        {"min": 3, "max": 3, "label": CATEGORY_LABELS[3], "color": CATEGORY_COLORS[3]},
        {"min": 4, "max": 4, "label": CATEGORY_LABELS[4], "color": CATEGORY_COLORS[4]},
    ]

    c = (
        Map(init_opts=opts.InitOpts(width="1500px", height="1050px", bg_color="white"))
        .add(
            series_name="聚类类别",
            data_pair=data_pair,
            maptype="china",
            is_map_symbol_show=False,
            label_opts=opts.LabelOpts(
                is_show=True,
                font_size=11,
                font_family="Microsoft YaHei",
                color="#222222",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=TITLE,
                subtitle=SUBTITLE,
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(
                    font_family="Microsoft YaHei",
                    font_size=26,
                    font_weight="bold",
                    color="#111111",
                ),
                subtitle_textstyle_opts=opts.TextStyleOpts(
                    font_family="Microsoft YaHei",
                    font_size=14,
                    color="#444444",
                ),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                is_piecewise=True,
                pieces=pieces,
                pos_left="3%",
                pos_top="20%",
                item_width=30,
                item_height=18,
                textstyle_opts=opts.TextStyleOpts(
                    font_family="Microsoft YaHei",
                    font_size=14,
                    color="#222222",
                ),
            ),
        )
        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(
                border_color="#666666",
                border_width=0.9,
                area_color="#F4F4F4",
            )
        )
    )

    # 始终先输出 HTML，保证至少可交付可视化结果
    html_path = c.render(str(OUTPUT_HTML))
    print(f"HTML 文件已保存：{OUTPUT_HTML.resolve()}")

    # 导出高分辨率 PNG：增加延迟，避免 ECharts JS 尚未加载完成
    try:
        make_snapshot(
            snapshot,
            html_path,
            str(OUTPUT_PNG),
            is_remove_html=False,
            pixel_ratio=3,
            delay=5,
        )
        print(f"PNG 图片已保存：{OUTPUT_PNG.resolve()}")
    except Exception as e:
        print("PNG 导出失败，原因可能是 ECharts JS 文件未加载成功。")
        print(f"错误信息：{e}")
        print(f"已保留 HTML 文件：{OUTPUT_HTML.resolve()}")
        print("请用浏览器打开该 HTML 文件后手动截图，或检查网络是否能访问 https://assets.pyecharts.org/assets/v5/")


if __name__ == "__main__":
    main()
