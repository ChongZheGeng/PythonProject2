#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制四次人口普查省级样本关键特征成对关系图。"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STANDARDIZED_FILE = "人口普查数据_格式修正版.xlsx"
DEFAULT_SHEET = "可视化主表"
TARGET_YEARS = [1990, 2000, 2010, 2020]
EXCLUDED_REGIONS = {"全国", "合计", "大陆合计"}

RAW_FILES = {
    1990: "第四次人口普查_省级聚类特征初筛总表_补充省际迁入迁出.xlsx",
    2000: "第五次人口普查_处理后_重庆并入四川_含迁移流动.xlsx",
    2010: "第六次人口普查_初筛总表_重庆并入四川_含迁移流动.xlsx",
    2020: "第七次人口普查_初筛汇总_重庆并入四川_含迁移流动.xlsx",
}


def resolve_file_path(file_name: str) -> Optional[Path]:
    """按“当前工作目录 -> 脚本目录”顺序查找文件。"""
    p = Path(file_name)
    if p.is_absolute() and p.exists():
        return p

    cwd_path = Path.cwd() / file_name
    if cwd_path.exists():
        return cwd_path

    script_path = Path(__file__).resolve().parent / file_name
    if script_path.exists():
        return script_path

    return None


def get_first_existing_column(df: pd.DataFrame, candidate_names: List[str]) -> Optional[str]:
    for name in candidate_names:
        if name in df.columns:
            return name
    return None


def normalize_ratio(series: pd.Series) -> pd.Series:
    """将比例列统一为0-1小数；中位数>1.5视作百分数。"""
    if series is None:
        return pd.Series(dtype=float)

    s = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    values = pd.to_numeric(s, errors="coerce")

    valid = values.dropna()
    if not valid.empty and valid.median() > 1.5:
        values = values / 100.0
    return values


def load_standardized_data(file_path: Path, sheet_name: str) -> pd.DataFrame:
    print(f"当前读取文件：{file_path}")
    print(f"当前读取工作表：{sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("当前字段列表：")
    print(list(df.columns))
    return df


def _col_candidates() -> Dict[str, List[str]]:
    return {
        "census_year": ["census_year", "普查年份", "年份"],
        "census_round": ["census_round", "普查轮次", "普查"],
        "region": ["region_short", "region", "地区", "地区简称"],
        "total_population": ["total_pop", "total_population", "总人口"],
        "urbanization_rate": ["urban_rate", "urbanization_rate", "城镇化率"],
        "city_share": ["city_share", "市占比", "市占总人口比", "市人口占总人口比重(%)", "市占总人口比重", "市人口占比"],
        "town_share": ["town_share", "镇占比", "镇占总人口比", "镇人口占总人口比重(%)", "镇占总人口比重", "镇人口占比"],
        "higher_edu_share": ["higher_edu_share", "higher_education_share", "college_junior_share", "bachelor_share", "graduate_share", "大学专科占比", "大学本科占比", "研究生占比", "高等教育占比"],
        "young_share": ["age_0_14_share", "0-14岁占比", "0-14岁占总人口比", "0-14岁占总人口比重(%)"],
        "old_share": [
            "age_65_plus_share", "age_60_plus_share", "65岁及以上占比", "65岁及以上占总人口比", "65岁及以上占总人口比重(%)", "60岁及以上占比", "60岁及以上占总人口比"
        ],
        "aging_coeff": ["aging_coeff", "老龄化系数"],
        "net_migration_rate": ["net_migration_rate", "net_interprovincial_inflow_rate", "净迁移率", "省际净迁入率"],
        "in_migration_share": ["interprovincial_inflow_share", "in_migration_share", "cross_province_in_share", "跨省迁入占总人口比例", "跨省迁入占总人口比", "跨省迁入占总人口比重(%)", "跨省流入占总人口比"],
        "out_migration_share": ["interprovincial_outflow_share", "out_migration_share", "cross_province_out_share", "跨省迁出占总人口比例", "跨省迁出占总人口比", "跨省迁出占总人口比重(%)", "跨省流出占总人口比"],
        "illiteracy_rate": ["illiteracy_rate", "文盲率", "文盲人口占15岁及以上人口比", "文盲人口占15岁及以上人口比重(%)"],
    }


def build_features_from_standardized(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
    cand = _col_candidates()
    mapped: Dict[str, str] = {}
    missing: Dict[str, List[str]] = {}

    for k, names in cand.items():
        col = get_first_existing_column(df, names)
        if col:
            mapped[k] = col
        else:
            missing[k] = names

    out = pd.DataFrame(index=df.index)
    derived_logs: List[str] = []

    for req in ["census_year", "region"]:
        if req not in mapped:
            raise ValueError(f"缺少关键字段：{req}，候选字段：{cand[req]}")

    out["census_year"] = pd.to_numeric(df[mapped["census_year"]], errors="coerce")
    out["region"] = df[mapped["region"]].astype(str).str.strip()

    out["census_round"] = df[mapped["census_round"]] if "census_round" in mapped else np.nan

    if "total_population" in mapped:
        out["total_population"] = pd.to_numeric(df[mapped["total_population"]], errors="coerce")

    if "urbanization_rate" in mapped:
        out["urbanization_rate"] = normalize_ratio(df[mapped["urbanization_rate"]])
    else:
        city_col = mapped.get("city_share")
        town_col = mapped.get("town_share")
        if city_col and town_col:
            out["urbanization_rate"] = normalize_ratio(df[city_col]) + normalize_ratio(df[town_col])

    if "higher_edu_share" in mapped:
        col = mapped["higher_edu_share"]
        if col in {"college_junior_share", "bachelor_share", "graduate_share", "大学专科占比", "大学本科占比", "研究生占比"}:
            mapped.pop("higher_edu_share", None)
        else:
            out["higher_edu_share"] = normalize_ratio(df[col])
            derived_logs.append(f"higher_edu_share = {col}")
    if "higher_edu_share" not in out.columns:
        college_col = get_first_existing_column(df, ["college_junior_share", "大学专科占比"])
        bachelor_col = get_first_existing_column(df, ["bachelor_share", "大学本科占比"])
        graduate_col = get_first_existing_column(df, ["graduate_share", "研究生占比"])
        if college_col and bachelor_col:
            college = normalize_ratio(df[college_col]).fillna(0)
            bachelor = normalize_ratio(df[bachelor_col]).fillna(0)
            graduate = normalize_ratio(df[graduate_col]).fillna(0) if graduate_col else 0
            out["higher_edu_share"] = college + bachelor + graduate
            expr = f"higher_edu_share = {college_col} + {bachelor_col}"
            if graduate_col:
                expr += f" + {graduate_col}"
            else:
                expr += " + 0(graduate_share缺失)"
            derived_logs.append(expr)
    else:
        pass

    if "aging_coeff" in mapped:
        out["aging_coeff"] = pd.to_numeric(df[mapped["aging_coeff"]], errors="coerce")
        derived_logs.append(f"aging_coeff = {mapped['aging_coeff']}")
    else:
        young_col = mapped.get("young_share")
        old_col = mapped.get("old_share")
        if young_col and old_col:
            young = normalize_ratio(df[young_col])
            old = normalize_ratio(df[old_col])
            # 1990年若只有60岁及以上占比，也可作为四普近似老年人口占比。
            out["aging_coeff"] = old / young.replace(0, np.nan)
            out["young_share"] = young
            out["old_share"] = old
            derived_logs.append(f"aging_coeff = {old_col} / {young_col}")

    if "net_migration_rate" in mapped:
        out["net_migration_rate"] = normalize_ratio(df[mapped["net_migration_rate"]])
        derived_logs.append(f"net_migration_rate = {mapped['net_migration_rate']}")
    else:
        in_col = mapped.get("in_migration_share")
        out_col = mapped.get("out_migration_share")
        if in_col and out_col:
            out["net_migration_rate"] = normalize_ratio(df[in_col]) - normalize_ratio(df[out_col])
            derived_logs.append(f"net_migration_rate = {in_col} - {out_col}")

    if "illiteracy_rate" in mapped:
        out["illiteracy_rate"] = normalize_ratio(df[mapped["illiteracy_rate"]])

    if "total_population" in out.columns:
        out["log_total_population"] = np.log10(out["total_population"].where(out["total_population"] > 0))

    out = out[~out["region"].isin(EXCLUDED_REGIONS)].copy()
    out = out[out["census_year"].isin(TARGET_YEARS)].copy()

    years_present = set(out["census_year"].dropna().astype(int).unique().tolist())
    missing_years = [y for y in TARGET_YEARS if y not in years_present]
    if missing_years:
        print(f"warning: 以下年份没有可用数据：{missing_years}")

    print("成功识别的字段映射：")
    for k, v in mapped.items():
        print(f"  {k} <- {v}")

    unresolved = {k: v for k, v in missing.items() if k not in mapped}
    print("未识别的字段：")
    if unresolved:
        for k, v in unresolved.items():
            print(f"  {k}: 候选={v}")
    else:
        print("  无")
    print("成功生成的派生变量：")
    if derived_logs:
        for item in derived_logs:
            print(f"  {item}")
    else:
        print("  无")

    return out, mapped, unresolved


def plot_pairplot(df: pd.DataFrame, mode: str, output_path: str) -> None:
    basic_vars = ["urbanization_rate", "higher_edu_share", "aging_coeff", "net_migration_rate"]
    extended_vars = basic_vars + ["log_total_population", "illiteracy_rate"]
    vars_to_plot = basic_vars if mode == "basic" else extended_vars

    candidate_hint = {
        "urbanization_rate": _col_candidates()["urbanization_rate"] + _col_candidates()["city_share"] + _col_candidates()["town_share"],
        "higher_edu_share": _col_candidates()["higher_edu_share"],
        "aging_coeff": _col_candidates()["aging_coeff"] + _col_candidates()["young_share"] + _col_candidates()["old_share"],
        "net_migration_rate": _col_candidates()["net_migration_rate"] + _col_candidates()["in_migration_share"] + _col_candidates()["out_migration_share"],
        "log_total_population": _col_candidates()["total_population"],
        "illiteracy_rate": _col_candidates()["illiteracy_rate"],
    }

    for v in vars_to_plot:
        if v not in df.columns or df[v].dropna().empty:
            raise ValueError(f"变量整列无法生成：{v}。请检查候选字段：{candidate_hint.get(v, [])}")

    label_map = {
        "urbanization_rate": "城镇化率",
        "higher_edu_share": "高等教育占比",
        "aging_coeff": "老龄化系数",
        "net_migration_rate": "净迁移率",
        "log_total_population": "log10(总人口)",
        "illiteracy_rate": "文盲率",
    }

    plot_df = df[["census_year", *vars_to_plot]].dropna().copy()
    if plot_df.empty:
        raise ValueError("绘图数据为空：请检查字段缺失或筛选条件。")
    print(f"绘图样本量：{len(plot_df)}")
    print("各年份样本量：")
    year_counts = plot_df["census_year"].value_counts().to_dict()
    for y in TARGET_YEARS:
        print(f"{y}: {int(year_counts.get(y, 0))}")

    plot_df = plot_df.rename(columns={k: v for k, v in label_map.items() if k in plot_df.columns})
    plot_vars_cn = [label_map[v] for v in vars_to_plot]
    plot_df["census_year"] = plot_df["census_year"].astype(int).astype(str)

    sns.set_theme(style="white", context="notebook")
    palette = {"1990": "#6BAED6", "2000": "#FDAE6B", "2010": "#74C476", "2020": "#C994C7"}

    g = sns.pairplot(
        plot_df,
        vars=plot_vars_cn,
        hue="census_year",
        palette=palette,
        diag_kind="kde",
        height=2.9,
        plot_kws={"s": 45, "alpha": 0.75, "edgecolor": "white", "linewidth": 0.7},
    )

    if g._legend is not None:
        g._legend.set_title("普查年份")

    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle("四次人口普查关键特征成对关系图", fontsize=15)
    g.fig.text(0.5, 0.965, "基于四普（1990）、五普（2000）、六普（2010）、七普（2020）省级样本", ha="center", fontsize=11)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(g.fig)
    print(f"已保存图像：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制人口普查多变量成对关系图")
    parser.add_argument("--file", default=STANDARDIZED_FILE, help="标准化Excel文件路径")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="工作表名称")
    parser.add_argument("--mode", choices=["basic", "extended"], default="basic", help="basic=4变量，extended=6变量")
    parser.add_argument("--output", default="outputs/census_pairplot_key_features.png", help="输出图像路径")
    args = parser.parse_args()

    resolved = resolve_file_path(args.file)
    if resolved is None:
        if args.file == STANDARDIZED_FILE:
            print(f"未找到标准化文件：{STANDARDIZED_FILE}\n请将该文件放在项目根目录，或使用 --file 指定完整路径。")
        else:
            print(f"未找到指定文件：{args.file}")
        return

    try:
        raw_df = load_standardized_data(resolved, args.sheet)
        feature_df, _, _ = build_features_from_standardized(raw_df)
        plot_pairplot(feature_df, mode=args.mode, output_path=args.output)
    except Exception as exc:
        print(f"运行失败：{exc}")


if __name__ == "__main__":
    main()
