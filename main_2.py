from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import MultiIndex
from scipy import stats
import matplotlib.dates as mdates

MISSING_THRESHOLD_PCT = 5.0
MONTHS_PER_YEAR = 12
AMOUNT_OF_YEARS_RECORD = 3

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
    if pd.api.types.is_numeric_dtype(s):
        return -1
    if pd.api.types.is_datetime64_any_dtype(s):
        return np.nan
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
    orders_df = fix_dependence_id(orders_df, dup_customer_id_df, "customer")

    data = Datasets(
        items=items_df,
        customers=customer_df,
        orders=orders_df,
        order_item=order_item_df
    )
    return data


def remove_row_from_linked_df(removed_ids: list, linked_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    fixed_df = linked_df[~linked_df[id_col].isin(removed_ids)]
    return fixed_df


def remove_missing_threshold(df: pd.DataFrame, name: str, id_col: str) -> tuple:
    df = df.dropna(subset=MANDATORY_COLUMNS[name]).copy()
    pct = df.isna().mean() * 100

    cols_over_threshold = [c for c in df.columns if pct[c] > MISSING_THRESHOLD_PCT]
    cols_under_threshold = [c for c in df.columns if 0 < pct[c] <= MISSING_THRESHOLD_PCT]
    removed_ids_df = df[df[cols_under_threshold].isna().any(axis=1)].set_index(id_col)
    removed_ids = removed_ids_df.index.tolist()
    df = df.dropna(subset=cols_under_threshold)

    for col in cols_over_threshold:
        df[col] = df[col].fillna(default_value(df[col]))
    return df, removed_ids

def handle_missing(data: Datasets) -> Datasets:
    items_df, items_removed_id = remove_missing_threshold(data.items, "items", "id")
    order_item_df = remove_row_from_linked_df(items_removed_id, data.order_item, "item_id")

    customers_df, customers_removed_id = remove_missing_threshold(data.customers, "customers", "id")
    orders_df = remove_row_from_linked_df(customers_removed_id, data.orders, "customer_id")

    orders_df, _ = remove_missing_threshold(orders_df, "orders", "order_id")
    order_item_df, _ = remove_missing_threshold(order_item_df, "order_item", "id")

    data = Datasets(
        items=items_df,
        customers=customers_df,
        orders=orders_df,
        order_item=order_item_df
    )
    return data


def get_joining_by_year(customers_df: pd.DataFrame) -> pd.Series:
    joining_by_year = customers_df.groupby(customers_df["joining_date"].dt.year).size()
    joining_by_year.index = joining_by_year.index.astype(int)
    return joining_by_year


def customers_by_gender(customers_df: pd.DataFrame) -> None:
    counts = customers_df["gender"].value_counts().rename_axis("gender").reset_index(name="amount_of_customers")
    print("Customers by gender")
    print(counts)
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


def _monthly_join_counts(customers_df: pd.DataFrame) -> pd.DataFrame:
    joining_date = customers_df["joining_date"]
    return (
        joining_date.groupby([joining_date.dt.year, joining_date.dt.month])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, MONTHS_PER_YEAR + 1), fill_value=0)
    )


def _plot_monthly_joins(customers_df: pd.DataFrame) -> None:
    joins_by_period = (customers_df["joining_date"].dt.to_period("M").value_counts().sort_index())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        joins_by_period.index.to_timestamp(),
        joins_by_period.values,
        marker="o",
        label="Monthly joins",
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=45)

    ax.set_xlabel("Timeline")
    ax.set_ylabel("Joins")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    show("customers_join_by_month")


def find_monthly_joining_trend(customers_df: pd.DataFrame) -> list[int]:
    _plot_monthly_joins(customers_df)
    monthly = _monthly_join_counts(customers_df)
    yearly_monthly_avg = monthly.sum(axis=1) / MONTHS_PER_YEAR
    years_above_avg = monthly.gt(yearly_monthly_avg, axis=0).sum()

    strong_months = years_above_avg[years_above_avg >= AMOUNT_OF_YEARS_RECORD].index.tolist()
    for month in strong_months:
        print(f"Month {month} consistently sees higher-than-average new customers.")

    return strong_months


def items_by_category(items_df: pd.DataFrame) -> None:
    counts = items_df["item_category"].value_counts().rename_axis("item_category").reset_index(name="count")
    print("Items by category")
    print(counts)
    threshold = counts["count"].sum() * 0.05
    counts.loc[counts["count"] <= threshold, "item_category"] = "other"
    counts = counts.groupby("item_category", as_index=False)["count"].sum()

    plt.pie(counts["count"], labels=counts["item_category"], autopct="%1.1f%%", startangle=90)
    plt.title("Items by category")
    show("items_by_category")


def get_highest_and_lowest_in_stock(items_df: pd.DataFrame) -> pd.DataFrame:
    clean_df = items_df.loc[items_df["stock_quantity"] >= 0]
    df = clean_df.loc[[clean_df['stock_quantity'].idxmax(), clean_df['stock_quantity'].idxmin()]]
    return df


def get_mean_quantity(items_df: pd.DataFrame) -> float:
    clean_df = items_df.loc[items_df["stock_quantity"] >= 0]
    return clean_df["stock_quantity"].mean()


def orders_by_customer (customers_df: pd.DataFrame) -> None:
    per_customer = customers_df["customer_id"].value_counts()
    dist = per_customer.value_counts().sort_index()
    ax = dist.plot.bar(color="steelblue", figsize=(8, 5), rot=0)
    ax.bar_label(ax.containers[0], padding=3)
    ax.set_title("How many orders do customers place?")
    ax.set_xlabel("orders per customer")
    ax.set_ylabel("number of customers")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    show("orders_by_customer")


def get_top_n_customers(orders_df: pd.DataFrame, customers_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    top_customers_ids = orders_df["customer_id"].value_counts().sort_values(ascending=False).head(n).index
    top_customers_df = customers_df.loc[customers_df["id"].isin(top_customers_ids), ["id", "first_name", "last_name"]]
    return top_customers_df.set_index("id").loc[top_customers_ids].reset_index()


def payment_method_by_customer(customers_df: pd.DataFrame) -> None:
    unique_counts = customers_df.groupby('payment_method')['customer_id'].nunique().reset_index(name='unique_customer_count')
    unique_counts.plot.bar(x="payment_method", y="unique_customer_count", legend=False, color="steelblue")
    plt.title("What payment method is most popular?"); plt.xlabel("payment method"); plt.ylabel("customers"); plt.xticks(rotation=45)
    show("payment_method_by_customer")


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
    data = handle_missing(data)
    return data

def analyze_customers(customers_df: pd.DataFrame) -> None:
    customers_by_gender(customers_df)
    age_distribution(customers_df)
    top_year, highest_count = get_best_joining_year(customers_df)
    yearly_joining_trend = find_yearly_joining_trend(customers_df)
    print(f"The best signup year was {int(top_year)} with {highest_count} signups!")
    print(f"Year by year joining trajectory:\n{yearly_joining_trend}")
    find_monthly_joining_trend(customers_df)

def analyze_items(items_df: pd.DataFrame) -> None:
    items_by_category(items_df)
    print("highest and lowest in stock")
    highest_and_lowest_in_stock_df = get_highest_and_lowest_in_stock(items_df)
    print(highest_and_lowest_in_stock_df.to_string())
    mean_quantity = get_mean_quantity(items_df)
    print(f"mean quantity: {mean_quantity}")


def analyze_orders(orders_df: pd.DataFrame, customers_df: pd.DataFrame) -> None:
    orders_by_customer(orders_df)
    top_n_customers_df = get_top_n_customers(orders_df, customers_df)
    print(f"Top {len(top_n_customers_df)} customers")
    print(top_n_customers_df.to_string())
    payment_method_by_customer(orders_df)


def main(data_dir: Path):
    data = load_data(data_dir)
    data = prepare_data(data)
    #analyze_customers(data.customers)
    #analyze_items(data.items)
    #analyze_orders(data.orders, data.customers)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR)