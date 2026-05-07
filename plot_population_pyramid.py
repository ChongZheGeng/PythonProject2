import argparse
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

TARGET_SHEETS = [
    "年龄金字塔数据",
    "age_sex_pyramid",
    "population_pyramid",
]

AGE_ORDER = [
    "0岁",
    "1-4岁",
    "5-9岁",
    "10-14岁",
    "15-19岁",
    "20-24岁",
    "25-29岁",
    "30-34岁",
    "35-39岁",
    "40-44岁",
    "45-49岁",
    "50-54岁",
    "55-59岁",
    "60-64岁",
    "65-69岁",
    "70-74岁",
    "75-79岁",
    "80-84岁",
    "85-89岁",
    "90-94岁",
    "95-99岁",
    "100岁及以上",
]

ERROR_MSG_MISSING_PYRAMID = (
    "当前 Excel 未包含按年龄段分性别的人口金字塔数据，请先补充 age_group、male_pop、female_pop 字段。"
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名并兼容中英文常见命名。"""
    aliases = {
        "census_year": ["census_year", "year", "年份", "普查年份", "普查年"],
        "region_short": ["region_short", "region", "地区", "地区简称", "省份"],
        "age_group": ["age_group", "age", "年龄段", "年龄组", "年龄"],
        "male_pop": ["male_pop", "male", "male_population", "男性人口", "男"],
        "female_pop": ["female_pop", "female", "female_population", "女性人口", "女"],
        "sex": ["sex", "gender", "性别"],
        "population": ["population", "pop", "人数", "人口", "人口数"],
    }

    normalized = {str(col).strip().lower(): col for col in df.columns}
    rename_map = {}

    for target, keys in aliases.items():
        for key in keys:
            source = normalized.get(str(key).strip().lower())
            if source is not None:
                rename_map[source] = target
                break

    return df.rename(columns=rename_map)


def _to_wide_from_long(df: pd.DataFrame) -> pd.DataFrame:
    required = {"census_year", "region_short", "age_group", "sex", "population"}
    if not required.issubset(df.columns):
        raise ValueError(ERROR_MSG_MISSING_PYRAMID)

    sex_map = {
        "男": "male",
        "男性": "male",
        "male": "male",
        "m": "male",
        "女": "female",
        "女性": "female",
        "female": "female",
        "f": "female",
    }

    tmp = df.copy()
    tmp["sex"] = tmp["sex"].astype(str).str.strip().str.lower().map(sex_map)
    tmp = tmp[tmp["sex"].isin(["male", "female"])]

    if tmp.empty:
        raise ValueError(ERROR_MSG_MISSING_PYRAMID)

    wide = (
        tmp.pivot_table(
            index=["census_year", "region_short", "age_group"],
            columns="sex",
            values="population",
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    if "male" not in wide.columns or "female" not in wide.columns:
        raise ValueError(ERROR_MSG_MISSING_PYRAMID)

    wide = wide.rename(columns={"male": "male_pop", "female": "female_pop"})
    return wide


def _format_data_structure_help(sheet_names: List[str], sheet_columns: List[str]) -> str:
    return (
        f"{ERROR_MSG_MISSING_PYRAMID}\n"
        f"当前 Excel 工作表：{', '.join(sheet_names) if sheet_names else '（无）'}\n"
        f"当前读取工作表字段：{', '.join(sheet_columns) if sheet_columns else '（无）'}\n"
        "期望数据格式示例：\n"
        "census_year | region_short | age_group | male_pop | female_pop\n"
        "2020        | 全国          | 0岁       | ...      | ...\n"
        "2020        | 全国          | 1-4岁     | ...      | ..."
    )


def resolve_input_file(file_path: str) -> str:
    """优先使用当前路径，若不存在则尝试脚本目录。"""
    if os.path.exists(file_path):
        return file_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, file_path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        "未找到 Excel 文件，请确认文件是否放在项目根目录，或使用 --file 指定完整路径。"
    )


def load_pyramid_data(file_path: str) -> pd.DataFrame:
    """读取金字塔数据，优先从指定 sheet 读取并兼容宽表/长表。"""
    resolved_file = resolve_input_file(file_path)

    try:
        excel = pd.ExcelFile(resolved_file)
    except Exception as exc:
        raise ValueError(f"Excel 文件读取失败：{exc}") from exc

    sheet_names = excel.sheet_names
    target_sheet = next((s for s in TARGET_SHEETS if s in sheet_names), None)
    if target_sheet is None:
        raise ValueError(_format_data_structure_help(sheet_names, []))

    df = pd.read_excel(resolved_file, sheet_name=target_sheet)
    raw_columns = [str(col) for col in df.columns]
    df = normalize_columns(df)

    wide_required = {"census_year", "region_short", "age_group", "male_pop", "female_pop"}
    if wide_required.issubset(df.columns):
        out = df[list(wide_required)].copy()
    else:
        try:
            out = _to_wide_from_long(df)
        except ValueError as exc:
            raise ValueError(_format_data_structure_help(sheet_names, raw_columns)) from exc

    for col in ["male_pop", "female_pop"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[["male_pop", "female_pop"]].isna().all().any():
        raise ValueError("人口字段不是数字，请检查 male_pop / female_pop 或 population 列。")

    out = out.dropna(subset=["male_pop", "female_pop"], how="all")
    out["census_year"] = pd.to_numeric(out["census_year"], errors="coerce")
    out = out.dropna(subset=["census_year", "region_short", "age_group"])
    out["census_year"] = out["census_year"].astype(int)
    out["region_short"] = out["region_short"].astype(str).str.strip()
    out["age_group"] = out["age_group"].astype(str).str.strip()

    return out


def filter_pyramid_data(df: pd.DataFrame, year: int, region: str) -> pd.DataFrame:
    filtered = df[(df["census_year"] == year) & (df["region_short"] == region)].copy()
    if filtered.empty:
        raise ValueError(f"指定年份或地区没有数据：year={year}, region={region}")
    return filtered


def prepare_plot_data(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    factor = 10000 if unit == "wan" else 1
    plot_df = df.copy()
    plot_df["male_plot"] = -(plot_df["male_pop"] / factor)
    plot_df["female_plot"] = plot_df["female_pop"] / factor

    order_map = {age: i for i, age in enumerate(AGE_ORDER)}
    plot_df["age_order"] = plot_df["age_group"].map(order_map)
    plot_df = plot_df.dropna(subset=["age_order"]).copy()

    if plot_df.empty:
        raise ValueError("筛选结果中没有可识别的年龄段，请检查 age_group 内容。")

    plot_df["age_order"] = plot_df["age_order"].astype(int)
    plot_df = plot_df.sort_values("age_order")
    return plot_df


def plot_population_pyramid(df: pd.DataFrame, year: int, region: str, unit: str, output_path: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    ax.barh(df["age_group"], df["male_plot"], color="#4E79A7", label="男性")
    ax.barh(df["age_group"], df["female_plot"], color="#E15759", label="女性")

    ax.axvline(x=0, color="black", linewidth=1)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="gray")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):,.0f}"))
    ax.set_title(f"{year}年{region}人口金字塔")
    ax.set_ylabel("年龄段")
    ax.set_xlabel("人数（万人）" if unit == "wan" else "人数")
    ax.legend()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 Excel 数据绘制人口金字塔图")
    parser.add_argument("--file", default="人口普查数据_格式修正版.xlsx", help="Excel 文件路径")
    parser.add_argument("--year", type=int, default=2020, help="普查年份，例如 2020")
    parser.add_argument("--region", default="全国", help="地区简称，例如 全国")
    parser.add_argument("--unit", choices=["raw", "wan"], default="wan", help="人数单位")
    parser.add_argument("--output", default=None, help="输出图片路径")
    return parser.parse_args(argv)


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.output:
        return args.output
    return os.path.join("outputs", f"population_pyramid_{args.year}_{args.region}.png")


def main() -> None:
    args = parse_args(sys.argv[1:])
    output_path = _resolve_output_path(args)

    print("当前绘图参数：")
    print(f"文件：{args.file}")
    print(f"年份：{args.year}")
    print(f"地区：{args.region}")
    print(f"单位：{'万人' if args.unit == 'wan' else '原始人数'}")
    print(f"输出路径：{output_path}")

    try:
        data = load_pyramid_data(args.file)
        filtered = filter_pyramid_data(data, args.year, args.region)
        plot_data = prepare_plot_data(filtered, args.unit)
        plot_population_pyramid(plot_data, args.year, args.region, args.unit, output_path)
        print(f"已保存人口金字塔图：{output_path}")
    except Exception as exc:
        print(f"错误：{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
