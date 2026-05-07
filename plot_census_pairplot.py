#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制四次人口普查省级样本关键特征成对关系图。"""

import argparse
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STANDARDIZED_FILE = "人口普查数据_格式修正版.xlsx"
RAW_FILES = {
    1990: "第四次人口普查_省级聚类特征初筛总表_补充省际迁入迁出.xlsx",
    2000: "第五次人口普查_处理后_重庆并入四川_含迁移流动.xlsx",
    2010: "第六次人口普查_初筛总表_重庆并入四川_含迁移流动.xlsx",
    2020: "第七次人口普查_初筛汇总_重庆并入四川_含迁移流动.xlsx",
}
ROUND_MAP = {1990: "四普", 2000: "五普", 2010: "六普", 2020: "七普"}


# 常见字段同义词映射（按优先级从上到下匹配）
CANDIDATES: Dict[str, List[str]] = {
    "region": [
        "地区",
        "省份",
        "省区市",
        "地区名称",
        "行政区",
        "名称",
    ],
    "urban_city": [
        "市占比",
        "市占总人口比",
        "市人口占总人口比重(%)",
        "市人口占总人口比重",
        "市人口比重",
    ],
    "urban_town": [
        "镇占比",
        "镇占总人口比",
        "镇人口占总人口比重(%)",
        "镇人口占总人口比重",
        "镇人口比重",
    ],
    "urban_total": ["市占比+镇占比", "市占总人口比+镇占总人口比"],
    "edu_junior": [
        "大学专科占比",
        "大学专科占6岁及以上人口比",
        "大学专科占6岁及以上人口比重(%)",
        "大学专科占3岁及以上人口比",
    ],
    "edu_bachelor": [
        "大学本科占比",
        "大学本科占6岁及以上人口比",
        "大学本科占6岁及以上人口比重(%)",
        "大学本科占3岁及以上人口比",
    ],
    "edu_master": [
        "研究生占6岁及以上人口比",
        "研究生占6岁及以上人口比重(%)",
        "硕士研究生占3岁及以上人口比",
    ],
    "edu_doctor": ["博士研究生占3岁及以上人口比"],
    "old65": ["65岁及以上占比", "65岁及以上人口占比", "65岁及以上人口比重(%)"],
    "old60": ["60岁及以上占比", "60岁及以上人口占比", "60岁及以上人口比重(%)"],
    "young014": ["0-14岁占比", "14岁及以下占比", "0-14岁人口占比", "0-14岁人口比重(%)"],
    "mig_in": [
        "跨省迁入占总人口比例",
        "跨省流入占总人口比",
        "跨省迁入占总人口比重(%)",
        "跨省迁入占总人口比",
    ],
    "mig_out": [
        "跨省迁出占总人口比例",
        "跨省流出占总人口比",
        "跨省迁出占总人口比重(%)",
        "跨省迁出占总人口比",
    ],
    "total_population": ["总人口", "常住人口", "人口总数"],
    "illiteracy_rate": ["文盲率", "文盲占比", "15岁及以上文盲率"],
}


def clean_col_name(col: str) -> str:
    return re.sub(r"\s+", "", str(col)).replace("（", "(").replace("）", ")")


def to_numeric_ratio(series: pd.Series) -> pd.Series:
    """将百分数/小数字段统一为小数比例。"""
    if series is None:
        return pd.Series(dtype=float)
    s = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    has_percent = s.str.contains("%", na=False)
    s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    num = pd.to_numeric(s, errors="coerce")

    if has_percent.any():
        num = num / 100.0
    else:
        # 启发式：若中位数明显 > 1，视为百分数
        med = num.dropna().median() if not num.dropna().empty else np.nan
        if pd.notna(med) and med > 1:
            num = num / 100.0
    return num


def locate_header_row(file_path: str, max_scan_rows: int = 20) -> int:
    """自动识别表头行。"""
    preview = pd.read_excel(file_path, header=None, nrows=max_scan_rows)
    for i in range(len(preview)):
        row_values = preview.iloc[i].astype(str).str.replace(r"\s+", "", regex=True)
        joined = "|".join(row_values.tolist())
        if any(k in joined for k in ["地区", "省份", "市占比", "总人口", "占比"]):
            return i
    return 0


def match_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    clean_map = {clean_col_name(c): c for c in columns}
    for key in candidates:
        key_clean = clean_col_name(key)
        for c_clean, c_raw in clean_map.items():
            if key_clean == c_clean or key_clean in c_clean:
                return c_raw
    return None


def read_excel_auto(file_path: str) -> pd.DataFrame:
    header_row = locate_header_row(file_path)
    return pd.read_excel(file_path, header=header_row)


def exclude_total_rows(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    mask = ~df[region_col].astype(str).str.contains("全国|合计|总计|汇总", na=False)
    return df.loc[mask].copy()


def standardize_features(df: pd.DataFrame, census_year: int) -> pd.DataFrame:
    cols = list(df.columns)
    region_col = match_column(cols, CANDIDATES["region"])
    if not region_col:
        raise ValueError(f"{census_year}年数据缺少地区字段，无法继续。")

    df = exclude_total_rows(df, region_col)
    out = pd.DataFrame()
    out["region"] = df[region_col].astype(str).str.strip()
    out["census_year"] = census_year
    out["census_round"] = ROUND_MAP[census_year]

    urban_total_col = match_column(cols, CANDIDATES["urban_total"])
    if urban_total_col:
        out["urbanization_rate"] = to_numeric_ratio(df[urban_total_col])
    else:
        city_col = match_column(cols, CANDIDATES["urban_city"])
        town_col = match_column(cols, CANDIDATES["urban_town"])
        if not city_col or not town_col:
            raise ValueError(f"{census_year}年数据缺少城镇化率所需字段（市占比/镇占比）。")
        out["urbanization_rate"] = to_numeric_ratio(df[city_col]) + to_numeric_ratio(df[town_col])

    edu_cols = []
    for key in ["edu_junior", "edu_bachelor", "edu_master", "edu_doctor"]:
        c = match_column(cols, CANDIDATES[key])
        if c:
            edu_cols.append(c)
    if len(edu_cols) < 2:
        raise ValueError(f"{census_year}年数据缺少高等教育占比所需字段。")
    out["higher_edu_share"] = sum(to_numeric_ratio(df[c]) for c in edu_cols)

    old65_col = match_column(cols, CANDIDATES["old65"])
    old60_col = match_column(cols, CANDIDATES["old60"])
    young_col = match_column(cols, CANDIDATES["young014"])
    if not young_col:
        raise ValueError(f"{census_year}年数据缺少0-14岁占比字段。")
    young = to_numeric_ratio(df[young_col])
    if old65_col:
        old = to_numeric_ratio(df[old65_col])
    elif old60_col and census_year == 1990:
        # 四普无65岁口径时，采用60岁及以上作为近似
        old = to_numeric_ratio(df[old60_col])
    else:
        raise ValueError(f"{census_year}年数据缺少老龄化系数分子字段（65+或四普60+）。")
    out["aging_coeff"] = old / young.replace(0, np.nan)

    in_col = match_column(cols, CANDIDATES["mig_in"])
    out_col = match_column(cols, CANDIDATES["mig_out"])
    if not in_col or not out_col:
        raise ValueError(f"{census_year}年数据缺少净迁移率所需字段（跨省迁入/迁出）。")
    out["net_migration_rate"] = to_numeric_ratio(df[in_col]) - to_numeric_ratio(df[out_col])

    pop_col = match_column(cols, CANDIDATES["total_population"])
    if pop_col:
        pop = pd.to_numeric(df[pop_col], errors="coerce")
        out["log_total_population"] = np.log(pop.where(pop > 0))

    ill_col = match_column(cols, CANDIDATES["illiteracy_rate"])
    if ill_col:
        out["illiteracy_rate"] = to_numeric_ratio(df[ill_col])

    return out


def load_year_data(file_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件：{file_path}")
    raw = read_excel_auto(file_path)
    return standardize_features(raw, year)


def load_census4_data() -> pd.DataFrame:
    return load_year_data(RAW_FILES[1990], 1990)


def load_census5_data() -> pd.DataFrame:
    return load_year_data(RAW_FILES[2000], 2000)


def load_census6_data() -> pd.DataFrame:
    return load_year_data(RAW_FILES[2010], 2010)


def load_census7_data() -> pd.DataFrame:
    return load_year_data(RAW_FILES[2020], 2020)


def combine_all_census_data() -> pd.DataFrame:
    # 优先尝试标准化总表
    if os.path.exists(STANDARDIZED_FILE):
        try:
            std_df = read_excel_auto(STANDARDIZED_FILE)
            out_frames = []
            for year in [1990, 2000, 2010, 2020]:
                sub = std_df[std_df.astype(str).apply(lambda r: r.str.contains(str(year), na=False)).any(axis=1)]
                if not sub.empty:
                    out_frames.append(standardize_features(sub.copy(), year))
            if len(out_frames) == 4:
                print(f"已使用标准化文件：{STANDARDIZED_FILE}")
                return pd.concat(out_frames, ignore_index=True)
            print("标准化文件字段不完整，回退到四个原始文件。")
        except Exception as e:
            print(f"标准化文件读取失败，回退原始文件。原因：{e}")

    frames = [load_census4_data(), load_census5_data(), load_census6_data(), load_census7_data()]
    return pd.concat(frames, ignore_index=True)


def plot_pairplot(df: pd.DataFrame, mode: str = "basic") -> None:
    os.makedirs("outputs", exist_ok=True)

    base_vars = ["urbanization_rate", "higher_edu_share", "aging_coeff", "net_migration_rate"]
    vars_to_plot = base_vars.copy()
    if mode == "extended":
        vars_to_plot += ["log_total_population", "illiteracy_rate"]

    missing = [c for c in vars_to_plot if c not in df.columns]
    if missing:
        raise ValueError(f"绘图所需字段缺失：{missing}")

    for c in vars_to_plot:
        miss_rate = df[c].isna().mean()
        if miss_rate > 0.3:
            warnings.warn(f"警告：字段 {c} 缺失比例较高（{miss_rate:.1%}）。")

    plot_df = df.dropna(subset=vars_to_plot + ["census_year"]).copy()
    plot_df["census_year"] = plot_df["census_year"].astype(str)

    sns.set_theme(style="white", context="notebook")
    palette = {"1990": "#4C78A8", "2000": "#F58518", "2010": "#54A24B", "2020": "#B279A2"}

    g = sns.pairplot(
        plot_df,
        vars=vars_to_plot,
        hue="census_year",
        palette=palette,
        diag_kind="kde",
        height=2.9,
        plot_kws={"s": 45, "alpha": 0.8, "edgecolor": "white", "linewidth": 0.7},
        diag_kws={"fill": True, "alpha": 0.3},
    )

    if g._legend is not None:
        g._legend.set_title("普查年份")

    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("四次人口普查关键特征成对关系图\n基于四普（1990）、五普（2000）、六普（2010）、七普（2020）省级样本", fontsize=14)

    out_path = "outputs/census_pairplot_key_features.png"
    g.savefig(out_path, dpi=320, bbox_inches="tight")
    print(f"已保存图像：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制人口普查多变量成对关系图")
    parser.add_argument("--mode", choices=["basic", "extended"], default="basic", help="basic=4变量，extended=6变量")
    args = parser.parse_args()

    df = combine_all_census_data()
    plot_pairplot(df, mode=args.mode)


if __name__ == "__main__":
    main()
