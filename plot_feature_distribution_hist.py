import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_chinese_font() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = None
    for name in candidates:
        if name in available_fonts:
            selected_font = name
            break

    if selected_font is None:
        print("警告：未找到常见中文字体，中文可能无法正常显示。")
        selected_font = "DejaVu Sans"
    else:
        print(f"当前使用中文字体：{selected_font}")

    mpl.rcParams["font.family"] = selected_font
    mpl.rcParams["font.sans-serif"] = [selected_font]
    mpl.rcParams["axes.unicode_minus"] = False

    sns.set_theme(
        style="white",
        font=selected_font,
        rc={
            "font.family": selected_font,
            "font.sans-serif": [selected_font],
            "axes.unicode_minus": False,
        },
    )
    return selected_font


def resolve_file_path(file_path: str) -> Path:
    input_path = Path(file_path)
    if input_path.exists():
        return input_path

    script_dir_path = Path(__file__).resolve().parent / file_path
    if script_dir_path.exists():
        return script_dir_path

    raise FileNotFoundError(f"未找到数据文件：{file_path}")


def get_first_existing_column(df: pd.DataFrame, candidate_names: list[str]) -> str | None:
    for col in candidate_names:
        if col in df.columns:
            return col
    return None


def normalize_ratio(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    if valid.median() > 1.5:
        return s / 100.0
    return s


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 年份
    year_col = get_first_existing_column(df, ["year", "年份", "普查年份"])
    if year_col is None:
        raise ValueError("缺少年份字段（候选：year, 年份, 普查年份）")
    out["year"] = pd.to_numeric(df[year_col], errors="coerce")

    # 地区名称（用于排除全国/合计）
    region_col = get_first_existing_column(df, ["region", "province", "地区", "省份", "省级行政区"])
    if region_col is not None:
        out["region_name"] = df[region_col].astype(str)
    else:
        out["region_name"] = ""

    if "region_type" in df.columns:
        out["region_type"] = df["region_type"].astype(str)
    else:
        out["region_type"] = ""

    # 1) 总人口
    col = get_first_existing_column(df, ["total_pop", "total_population", "总人口"])
    out["total_population"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    # 2) 城镇化率
    col = get_first_existing_column(df, ["urban_rate", "urbanization_rate", "城镇化率"])
    if col:
        out["urbanization_rate"] = df[col]
    else:
        city_col = get_first_existing_column(df, ["city_share", "市占比", "市占总人口比"])
        town_col = get_first_existing_column(df, ["town_share", "镇占比", "镇占总人口比"])
        if city_col and town_col:
            out["urbanization_rate"] = pd.to_numeric(df[city_col], errors="coerce") + pd.to_numeric(df[town_col], errors="coerce")
        else:
            out["urbanization_rate"] = pd.NA

    # 3) 高等教育占比
    higher_col = get_first_existing_column(df, ["higher_edu_share"])
    if higher_col:
        out["higher_edu_share"] = df[higher_col]
    else:
        junior_col = get_first_existing_column(df, ["college_junior_share", "大学专科占比", "大学专科占6岁及以上人口比", "大学专科占3岁及以上人口比"])
        bachelor_col = get_first_existing_column(df, ["bachelor_share", "大学本科占比", "大学本科占6岁及以上人口比", "大学本科占3岁及以上人口比"])
        graduate_col = get_first_existing_column(df, ["graduate_share", "研究生占比", "研究生占6岁及以上人口比", "硕士研究生占3岁及以上人口比"])
        phd_col = get_first_existing_column(df, ["博士研究生占3岁及以上人口比"])

        if junior_col and bachelor_col:
            junior = pd.to_numeric(df[junior_col], errors="coerce")
            bachelor = pd.to_numeric(df[bachelor_col], errors="coerce")
            graduate = pd.to_numeric(df[graduate_col], errors="coerce") if graduate_col else pd.Series(0, index=df.index)
            phd = pd.to_numeric(df[phd_col], errors="coerce") if phd_col else pd.Series(0, index=df.index)
            out["higher_edu_share"] = junior + bachelor + graduate.fillna(0) + phd.fillna(0)
        else:
            out["higher_edu_share"] = pd.NA

    # 4) 老龄化系数
    young_col = get_first_existing_column(df, ["age_0_14_share", "0-14岁占比", "0-14岁占总人口比"])
    age65_col = get_first_existing_column(df, ["age_65_plus_share", "65岁及以上占比"])
    age60_col = get_first_existing_column(df, ["age_60_plus_share", "60岁及以上占比"])

    out["age_0_14_share"] = pd.to_numeric(df[young_col], errors="coerce") if young_col else pd.NA
    out["age_65_plus_share"] = pd.to_numeric(df[age65_col], errors="coerce") if age65_col else pd.NA
    out["age_60_plus_share"] = pd.to_numeric(df[age60_col], errors="coerce") if age60_col else pd.NA

    old_share = out["age_65_plus_share"].copy()
    old_share = old_share.where(old_share.notna(), out["age_60_plus_share"])

    # 1990 年如果缺失 65+，允许使用 60+ 近似
    use_60_for_1990 = (
        (out["year"] == 1990)
        & out["age_65_plus_share"].isna()
        & out["age_60_plus_share"].notna()
    )
    if use_60_for_1990.any():
        print("说明：1990年部分样本缺少65岁及以上占比，已使用60岁及以上占比近似计算老龄化系数。")

    young = out["age_0_14_share"].replace(0, pd.NA)
    out["aging_coeff"] = old_share / young

    # 5) 净迁移率
    col = get_first_existing_column(df, ["net_interprovincial_inflow_rate", "net_migration_rate", "省际净迁入率", "净迁移率"])
    if col:
        out["net_migration_rate"] = df[col]
    else:
        inflow_col = get_first_existing_column(df, ["interprovincial_inflow_share", "跨省迁入占总人口比例", "跨省流入占总人口比"])
        outflow_col = get_first_existing_column(df, ["interprovincial_outflow_share", "跨省迁出占总人口比例", "跨省流出占总人口比"])
        if inflow_col and outflow_col:
            out["net_migration_rate"] = pd.to_numeric(df[inflow_col], errors="coerce") - pd.to_numeric(df[outflow_col], errors="coerce")
        else:
            out["net_migration_rate"] = pd.NA

    # 6) 文盲率
    col = get_first_existing_column(df, ["illiteracy_rate", "文盲率", "文盲人口占15岁及以上人口比", "文盲人口占15岁及以上人口比重(%)"])
    out["illiteracy_rate"] = df[col] if col else pd.NA

    # 比例字段归一化
    for ratio_col in [
        "urbanization_rate",
        "higher_edu_share",
        "age_0_14_share",
        "age_65_plus_share",
        "age_60_plus_share",
        "net_migration_rate",
        "illiteracy_rate",
    ]:
        out[ratio_col] = normalize_ratio(out[ratio_col])

    return out


def plot_feature_histograms(feature_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)

    plot_map = [
        ("total_population", "总人口"),
        ("urbanization_rate", "城镇化率"),
        ("higher_edu_share", "高等教育占比"),
        ("aging_coeff", "老龄化系数"),
        ("net_migration_rate", "净迁移率"),
        ("illiteracy_rate", "文盲率"),
    ]

    for ax, (col, cname) in zip(axes.flatten(), plot_map):
        values = pd.to_numeric(feature_df[col], errors="coerce").dropna()
        if values.empty:
            print(f"warning: 指标 {col} 无法生成有效数据，将在子图显示“数据缺失”。")
            ax.text(0.5, 0.5, "数据缺失", ha="center", va="center", fontsize=13)
            ax.set_title(cname)
            ax.set_xlabel(cname)
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.15)
            sns.despine(ax=ax)
            continue

        sns.histplot(
            values,
            bins=18,
            kde=True,
            color="#9EC3E6",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.75,
            ax=ax,
        )

        if ax.lines:
            ax.lines[-1].set_color("#0072BC")
            ax.lines[-1].set_linewidth(1.6)

        ax.set_title(cname)
        ax.set_xlabel(cname)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.15)
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle("四次人口普查关键特征分布直方图", fontsize=18, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.955,
        "基于四普（1990）、五普（2000）、六普（2010）、七普（2020）省级样本",
        ha="center",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.86, wspace=0.28, hspace=0.34)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存图像：{output.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制四次人口普查关键特征分布直方图")
    parser.add_argument("--file", default="人口普查数据_格式修正版.xlsx")
    parser.add_argument("--sheet", default="可视化主表")
    parser.add_argument("--output", default="outputs/census_feature_distribution_hist.png")
    parser.add_argument("--include-national", action="store_true", default=False)
    args = parser.parse_args()

    setup_chinese_font()

    input_file = resolve_file_path(args.file)
    print(f"当前读取文件：{input_file.as_posix()}")
    print(f"当前读取工作表：{args.sheet}")

    df = pd.read_excel(input_file, sheet_name=args.sheet)
    print(f"当前字段列表：{list(df.columns)}")

    feature_df = build_feature_dataframe(df)

    # 样本筛选
    filtered = feature_df[feature_df["year"].isin([1990, 2000, 2010, 2020])].copy()

    if not args.include_national:
        excluded_names = {"全国", "合计", "大陆合计"}
        filtered = filtered[~filtered["region_name"].isin(excluded_names)]

    if "region_type" in filtered.columns and (filtered["region_type"].astype(str).str.len() > 0).any():
        filtered = filtered[filtered["region_type"] == "province"]

    print("各年份样本量：")
    for y in [1990, 2000, 2010, 2020]:
        print(f"{y}: {int((filtered['year'] == y).sum())}")

    print("成功生成的指标：")
    for c in [
        "total_population",
        "urbanization_rate",
        "higher_edu_share",
        "aging_coeff",
        "net_migration_rate",
        "illiteracy_rate",
    ]:
        print(c)

    print("各指标非空样本量：")
    name_map = {
        "total_population": "总人口",
        "urbanization_rate": "城镇化率",
        "higher_edu_share": "高等教育占比",
        "aging_coeff": "老龄化系数",
        "net_migration_rate": "净迁移率",
        "illiteracy_rate": "文盲率",
    }
    for col, cname in name_map.items():
        print(f"{cname}: {int(pd.to_numeric(filtered[col], errors='coerce').notna().sum())}")

    plot_feature_histograms(filtered, args.output)


if __name__ == "__main__":
    main()
