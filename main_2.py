from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MISSING_THRESHOLD_PCT = 5.0

DATA_DIR = Path("data")
CHARTS_DIR = Path("charts")

MANDATORY_COLUMNS = {
    "items": ["id", "item_name"],
    "customers": ["id", "first_name", "last_name"],
    "orders": ["order_id", "customer_id"],
    "order_item": ["id", "order_id", "item_id", "quantity"],
}


@dataclass(frozen=True)
class Datasets:
    items: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    order_item: pd.DataFrame


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")


def show(chart_name: str) -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{chart_name}.png", dpi=120)
    plt.show()
    plt.close("all")


def default_value(s: pd.Series):
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return -1
    return "Unknown"


def clean_duplicate(df: pd.DataFrame, sort_by: str, id_col: str = "id") -> tuple[pd.DataFrame, pd.DataFrame]:
    result_df = df.copy()
    kept_col = id_col + "_kept"
    removed_col = id_col + "_removed"
    result_df["missing_count"] = result_df.isna().sum(axis=1)
    result_df = result_df.sort_values(by=[sort_by, "missing_count", id_col])
    result_df[kept_col] = result_df.groupby(sort_by)[id_col].transform("first")
    is_duplicate = result_df[id_col] != result_df[kept_col]
    cleaned_df = result_df[~is_duplicate].drop(columns=["missing_count", kept_col]).copy()
    removal_log_df = result_df[is_duplicate][[kept_col, id_col]].copy()
    removal_log_df = removal_log_df.rename(columns={id_col: removed_col})

    return cleaned_df, removal_log_df


def fix_dependence_id(df: pd.DataFrame, dup_id_df: pd.DataFrame, dup_df_name: str, id_col: str = "id") -> pd.DataFrame:
    df_fixed = df.copy()
    kept_col = id_col + "_kept"
    removed_col = id_col + "_removed"
    df_id = dup_df_name + "_" + id_col
    id_mapping = dict(zip(dup_id_df[removed_col], dup_id_df[kept_col]))
    df_fixed[df_id] = df_fixed[df_id].replace(id_mapping)
    return df_fixed


def normalize_types(data: Datasets) -> Datasets:
    customers_df = data.customers
    customers_df["joining_date"] = pd.to_datetime(customers_df["joining_date"], errors="coerce")

    return Datasets(
        items=data.items,
        customers=customers_df,
        orders=data.orders,
        order_item=data.order_item,
    )


def handle_duplicates(raw: Datasets) -> Datasets:
    data = Datasets(*(df.copy() for df in (raw.items, raw.customers, raw.orders, raw.order_item)))
    items_df = data.items
    order_item_df = data.order_item
    customer_df = data.customers
    orders_df = data.orders

    items_df["item_brand"] = items_df["item_brand"].replace(["", "empty", "None", "Unknown"], np.nan)
    items_df, dup_item_id_df = clean_duplicate(items_df, "item_name")
    order_item_df = fix_dependence_id(order_item_df, dup_item_id_df, "item")

    customer_df, dup_customer_id_df = clean_duplicate(customer_df, "email")
    orders_df = fix_dependence_id(orders_df, dup_item_id_df, "customer")

    data = Datasets(
        items=items_df,
        customers=customer_df,
        orders=orders_df,
        order_item=order_item_df
    )
    return data


def handel_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.dropna(subset=MANDATORY_COLUMNS[name]).copy()
    pct = df.isna().mean() * 100

    cols_over_threshold = [c for c in df.columns if pct[c] > MISSING_THRESHOLD_PCT]
    cols_under_threshold = [c for c in df.columns if 0 < pct[c] <= MISSING_THRESHOLD_PCT]
    df = df.dropna(subset=cols_under_threshold)

    for col in cols_over_threshold:
        df[col] = df[col].fillna(default_value(df[col]))
    return df


def get_joining_by_year(customers_df: pd.DataFrame) -> pd.Series:
    joining_by_year = customers_df.groupby(customers_df["joining_date"].dt.year).size()
    joining_by_year.index = joining_by_year.index.astype(int)
    return joining_by_year


def customers_by_gender(customers_df: pd.DataFrame) -> None:
    counts = customers_df["gender"].value_counts().rename_axis("gender").reset_index(name="amount_of_customers")
    counts.plot.bar(x="gender", y="amount_of_customers", legend=False, color="steelblue")
    plt.title("Customers by gender"); plt.ylabel("Amount of customers"); plt.xticks(rotation=45)
    show("customers_by_gender")


def age_distribution(customers: pd.DataFrame) -> None:
    customers["age"].plot.hist(bins=20, edgecolor="black", color="seagreen")
    plt.title("Customer age distribution"); plt.xlabel("Age")
    show("customer_age_distribution")


def get_best_joining_year(customers_df: pd.DataFrame) -> tuple:
    joining_by_year = get_joining_by_year(customers_df)
    top_year = joining_by_year.idxmax()
    highest_count = joining_by_year.max()
    return top_year, highest_count


def find_yearly_joining_trend(customers_df: pd.DataFrame) -> str:
    joining_by_year = get_joining_by_year(customers_df)
    joining_by_year.plot.bar(x="year", y="amount_of_joining", legend=False, color="steelblue")
    plt.title("Customers by year"); plt.xlabel("year"); plt.ylabel("Amount of customers"); plt.xticks(rotation=45)
    show("customers_joining_by_year")

    change = joining_by_year.pct_change().dropna()
    if (change > 0).all():
        return "Consistent Increase: Sign-ups grew steadily every single year."
    elif (change < 0).all():
        return "Consistent Decrease: Sign-ups dropped steadily every single year."
    else:
        return "Sporadic Pattern: The trajectory fluctuates year-by-year rather than moving in a single direction."


def find_monthly_joining_trend(customers_df: pd.DataFrame) -> str:
    monthly = customers_df.set_index("joining_date").resample("MS").size()



def load_data(data_dir: Path) -> Datasets:
    items_path = data_dir / "Items.csv"
    customers_path = data_dir / "customers.csv"
    orders_path = data_dir / "orders.csv"
    order_item_path = data_dir / "order_item.csv"

    require_file(items_path)
    require_file(customers_path)
    require_file(orders_path)
    require_file(order_item_path)

    data = Datasets(
        items=pd.read_csv(items_path),
        customers=pd.read_csv(customers_path),
        orders=pd.read_csv(orders_path),
        order_item = pd.read_csv(order_item_path)
    )

    return data

def prepare_data(raw: Datasets) -> Datasets:
    data = Datasets(*(df.copy() for df in (raw.items, raw.customers, raw.orders, raw.order_item)))
    data = normalize_types(data)
    data = handle_duplicates(data)
    data = Datasets(
        items=handel_missing(data.items, "items"),
        customers=handel_missing(data.customers, "customers"),
        orders=handel_missing(data.orders, "orders"),
        order_item=handel_missing(data.order_item, "order_item")
    )
    return data

def analyze_customers(customers_df: pd.DataFrame) -> None:
    """
    customers_by_gender(customers_df)
    age_distribution(customers_df)
    top_year, highest_count = get_best_joining_year(customers_df)
    print(f"The best signup year was {int(top_year)} with {highest_count} signups!")
    print("Year by year joining trajectory:\n"
          f"{find_yearly_joining_trend(customers_df)}")
    """
    print(find_monthly_joining_trend(customers_df))


def main(data_dir: Path):
    data = load_data(data_dir)
    data = prepare_data(data)
    analyze_customers(data.customers)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR)