from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FIG_DIR = Path("figs")
MISSING_THRESHOLD_PCT = 5.0
CAMPAIGN_MONTH = 8
USA_PATTERN = r"\b(?:usa|u\.s\.a?|united states|american|us)\b"
MANDATORY_COLUMNS = {
    "items": ["item_id", "item_name"],
    "customers": ["id", "first_name", "last_name"],
    "orders": ["order_id", "customer_id"],
    "order_item": ["id", "order_id", "item_id", "quantity"],
}

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


@dataclass(frozen=True)
class Datasets:
    items: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    order_item: pd.DataFrame


def heading(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def show(fig_name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fig_name}.png", dpi=120)
    plt.show()
    plt.close("all")


def load_data(data_dir: Path) -> Datasets:
    return Datasets(
        items=pd.read_csv(data_dir / "Items.csv"),
        customers=pd.read_csv(data_dir / "customers.csv"),
        orders=pd.read_csv(data_dir / "orders.csv"),
        order_item=pd.read_csv(data_dir / "order_item.csv"),
    )


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        cleaned = s.astype("string").str.strip().replace("", pd.NA).astype(object)
        df[col] = cleaned.where(cleaned.notna(), np.nan)
    return df


def parse_dates(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        day_first = pd.to_datetime(s, errors="coerce", dayfirst=True)
        month_first = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return month_first if month_first.isna().sum() < day_first.isna().sum() else day_first


def normalize_types(data: Datasets) -> Datasets:
    items, customers = clean_strings(data.items), clean_strings(data.customers)
    orders, order_item = clean_strings(data.orders), clean_strings(data.order_item)

    customers["joining_date"] = parse_dates(customers["joining_date"])
    orders["order_date"] = parse_dates(orders["order_date"])
    items["item_availability"] = (items["item_availability"].astype("string").str.lower().map({"true": True, "false": False}).astype("boolean"))
    for col in ("item_price", "stock_quantity"):
        items[col] = pd.to_numeric(items[col], errors="coerce")
    orders["delivery_days"] = pd.to_numeric(orders["delivery_days"], errors="coerce")
    order_item["quantity"] = pd.to_numeric(order_item["quantity"], errors="coerce")
    return Datasets(items, customers, orders, order_item)


def _keeper(group: pd.DataFrame, id_col: str) -> object:
    return group.sort_values(["_info", id_col], ascending=[False, True])[id_col].iloc[0]


def find_duplicate_items(items: pd.DataFrame) -> dict:
    df = items.assign(
        _name=items["item_name"].str.lower(),
        _brand=items["item_brand"].str.lower(),
        _info=items.notna().sum(axis=1),
    )
    mapping: dict = {}
    for _, group in df.dropna(subset=["_name"]).groupby("_name"):
        if len(group) < 2:
            continue
        keepers: dict = {}
        for brand, branded in group[group["_brand"].notna()].groupby("_brand"):
            keep = _keeper(branded, "item_id")
            keepers[brand] = keep
            mapping.update({rid: keep for rid in branded["item_id"] if rid != keep})

        brandless = group[group["_brand"].isna()]
        if brandless.empty:
            continue
        if keepers:
            target = _keeper(group[group["item_id"].isin(keepers.values())], "item_id")
            mapping.update({rid: target for rid in brandless["item_id"]})
        else:
            keep = _keeper(brandless, "item_id")
            mapping.update({rid: keep for rid in brandless["item_id"] if rid != keep})
    return mapping


def find_duplicate_customers(customers: pd.DataFrame) -> dict:
    df = customers.assign(_email=customers["email"].str.lower(), _info=customers.notna().sum(axis=1))
    mapping: dict = {}
    for _, group in df.dropna(subset=["_email"]).groupby("_email"):
        if len(group) < 2:
            continue
        keep = _keeper(group, "id")
        mapping.update({rid: keep for rid in group["id"] if rid != keep})
    return mapping


def remove_duplicates(data: Datasets) -> Datasets:
    item_map = find_duplicate_items(data.items)
    cust_map = find_duplicate_customers(data.customers)
    print(f"Duplicate items removed: {len(item_map)}")
    print(f"Duplicate customers removed: {len(cust_map)}")

    order_item = data.order_item.assign(item_id=data.order_item["item_id"].replace(item_map))
    orders = data.orders.assign(customer_id=data.orders["customer_id"].replace(cust_map))
    return Datasets(
        items=data.items[~data.items["item_id"].isin(item_map)],
        customers=data.customers[~data.customers["id"].isin(cust_map)],
        orders=orders,
        order_item=order_item,
    )


def default_value(s: pd.Series):
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s.median()
    return "Unknown"


def handle_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=MANDATORY_COLUMNS[name]).copy()
    pct = df.isna().mean() * 100
    print(f"\n[{name}] missing % per column (after mandatory-row removal):")
    print(pct[pct > 0].round(2).to_string() if (pct > 0).any() else "none")

    high = [c for c in df.columns if pct[c] > MISSING_THRESHOLD_PCT]
    low = [c for c in df.columns if 0 < pct[c] <= MISSING_THRESHOLD_PCT]
    for col in high:
        df[f"{col}_filled"] = df[col].isna()
        df[col] = df[col].fillna(default_value(df[col]))
    df = df.dropna(subset=low)
    print(f"[{name}] rows removed: {before - len(df)}, columns filled with default: {high}")
    return df


def enforce_consistency(data: Datasets) -> Datasets:
    orders = data.orders[data.orders["customer_id"].isin(data.customers["id"])]
    order_item = data.order_item[data.order_item["item_id"].isin(data.items["item_id"])]
    order_item = order_item[order_item["order_id"].isin(orders["order_id"])]
    print(f"\nOrders dropped (missing/unknown customer): {len(data.orders) - len(orders)}")
    print(f"Order lines dropped (missing/unknown item or order): {len(data.order_item) - len(order_item)}")
    return replace(data, orders=orders, order_item=order_item)


def add_order_totals(data: Datasets) -> Datasets:
    lines = data.order_item.merge(data.items[["item_id", "item_price"]], on="item_id", how="left")
    lines["line_total"] = lines["quantity"] * lines["item_price"]
    totals = lines.groupby("order_id").agg(total_price=("line_total", "sum"), n_items=("quantity", "sum"))
    orders = data.orders.drop(columns=["total_price", "n_items"], errors="ignore").merge(
        totals, left_on="order_id", right_index=True, how="left")
    return replace(data, orders=orders)


def prepare_data(raw: Datasets) -> Datasets:
    heading("Data preparation")
    data = Datasets(*(df.copy() for df in (raw.items, raw.customers, raw.orders, raw.order_item)))
    data = normalize_types(data)
    data = remove_duplicates(data)
    data = Datasets(
        items=handle_missing(data.items, "items"),
        customers=handle_missing(data.customers, "customers"),
        orders=handle_missing(data.orders, "orders"),
        order_item=handle_missing(data.order_item, "order_item"),
    )
    return enforce_consistency(data)


def customers_by_gender(customers: pd.DataFrame) -> None:
    counts = customers["gender"].value_counts().rename_axis("gender").reset_index(name="customers")
    print(counts)
    counts.plot.bar(x="gender", y="customers", legend=False, color="steelblue")
    plt.title("Customers by gender"); plt.ylabel("Customers"); plt.xticks(rotation=45)
    show("1a_gender")


def age_distribution(customers: pd.DataFrame) -> None:
    customers["age"].plot.hist(bins=20, edgecolor="black", color="seagreen")
    plt.title("Customer age distribution"); plt.xlabel("Age")
    show("1b_age")


def joining_year_and_trend(customers: pd.DataFrame) -> None:
    yearly = customers.groupby(customers["joining_date"].dt.year).size()
    print(yearly)
    print(f"Year with most joining customers: {yearly.idxmax()} ({yearly.max()} customers)")

    monthly = customers.set_index("joining_date").resample("MS").size()
    x = np.arange(len(monthly))
    fit = stats.linregress(x, monthly.values)
    print(f"\nYearly change (%):\n{(yearly.pct_change() * 100).round(1).to_string()}")
    print(f"\nMonthly trend: slope={fit.slope:.3f}/month, r={fit.rvalue:.2f}, p={fit.pvalue:.3f}")
    if fit.pvalue < 0.05:
        print("-> significant", "increase" if fit.slope > 0 else "decrease", "over time")
    else:
        print("-> no significant trend: sign-ups look sporadic / flat")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    yearly.plot.bar(ax=axes[0], color="steelblue", title="New customers per year")
    axes[1].plot(monthly.index, monthly.values, label="monthly joins")
    axes[1].plot(monthly.index, fit.intercept + fit.slope * x, "r--", label="linear trend")
    axes[1].set_title("New customers per month"); axes[1].legend()
    show("1d_trend")


def joining_seasonality(customers: pd.DataFrame) -> None:
    dates = customers["joining_date"]
    pivot = customers.groupby([dates.dt.year, dates.dt.month]).size().unstack(fill_value=0)
    avg_by_month = pivot.mean().round(1)
    top3_years = (pivot.rank(axis=1, ascending=False) <= 3).sum().rename("years_in_top3")
    print(pivot)
    print(pd.concat([avg_by_month.rename("avg_joins"), top3_years], axis=1))

    chi = stats.chisquare(pivot.sum().values)
    best = avg_by_month.idxmax()
    print(f"\nMonth totals vs uniform: chi2={chi.statistic:.1f}, p={chi.pvalue:.3f}")
    print(f"Highest average month: {best} (top-3 in {top3_years[best]} of {len(pivot)} years)")
    print("-> seasonality is real" if chi.pvalue < 0.05 else "-> no clear seasonality")

    avg_by_month.plot.bar(color="darkorange")
    plt.title("Average new customers per calendar month"); plt.xlabel("Month")
    show("1e_seasonality")


def analyze_customers(data: Datasets) -> None:
    heading("1. Customers")
    customers_by_gender(data.customers)
    age_distribution(data.customers)
    joining_year_and_trend(data.customers)
    joining_seasonality(data.customers)


def items_by_category(items: pd.DataFrame) -> None:
    counts = items["item_category"].value_counts().rename_axis("item_category").reset_index(name="items")
    print(counts)
    plt.pie(counts["items"], labels=counts["item_category"], autopct="%1.1f%%", startangle=90)
    plt.title("Items by category")
    show("2a_categories")


def stock_extremes_and_mean(items: pd.DataFrame) -> None:
    real = items[~items["stock_quantity_filled"]] if "stock_quantity_filled" in items else items
    top, bottom = real["stock_quantity"].max(), real["stock_quantity"].min()
    extremes = pd.concat([
        real[real["stock_quantity"] == top].assign(extreme="largest stock"),
        real[real["stock_quantity"] == bottom].assign(extreme="lowest stock")
    ])
    print(extremes[["extreme", "item_id", "item_name", "item_brand", "stock_quantity"]])
    print(f"\nMean stock quantity: {real['stock_quantity'].mean():.2f}")


def analyze_items(data: Datasets) -> None:
    heading("2. Items")
    items_by_category(data.items)
    stock_extremes_and_mean(data.items)


def purchases_per_customer(orders: pd.DataFrame) -> pd.Series:
    return orders.groupby("customer_id").size()


def purchase_distribution(orders: pd.DataFrame) -> None:
    dist = (purchases_per_customer(orders).value_counts().sort_index().rename_axis("purchases").reset_index(name="customers"))
    print(dist)
    dist.plot.bar(x="purchases", y="customers", legend=False, color="slateblue")
    plt.title("Customers by number of purchases"); plt.ylabel("Customers")
    show("3a_purchases")


def top_customers(orders: pd.DataFrame, customers: pd.DataFrame, n: int = 5) -> None:
    top = purchases_per_customer(orders).nlargest(n).rename("orders").reset_index()
    top = top.merge(customers[["id", "first_name", "last_name"]], left_on="customer_id", right_on="id")
    print(top[["customer_id", "first_name", "last_name", "orders"]])


def customers_per_payment_method(orders: pd.DataFrame) -> None:
    pay = orders.groupby("payment_method")["customer_id"].nunique().rename("customers").reset_index()
    print(pay)
    plt.pie(pay["customers"], labels=pay["payment_method"], autopct="%1.1f%%", startangle=90)
    plt.title("Customers per payment method (a customer can appear in several)")
    show("3c_payment")


def analyze_orders(data: Datasets) -> None:
    heading("3. Orders")
    purchase_distribution(data.orders)
    top_customers(data.orders, data.customers)
    customers_per_payment_method(data.orders)


def order_total_stats(orders: pd.DataFrame) -> None:
    print(orders[["order_id", "total_price"]].head())
    print(f"Orders without item lines (NaN total): {orders['total_price'].isna().sum()}")
    print(orders["total_price"].agg(["max", "min", "mean"]).round(2))


def items_vs_delivery(orders: pd.DataFrame) -> None:
    d = orders.dropna(subset=["n_items", "delivery_days"])
    pearson = stats.pearsonr(d["n_items"], d["delivery_days"])
    spearman = stats.spearmanr(d["n_items"], d["delivery_days"])
    print(f"Pearson r={pearson[0]:.3f} (p={pearson[1]:.3f}), Spearman rho={spearman[0]:.3f} (p={spearman[1]:.3f})")
    if pearson[1] >= 0.05:
        print("-> no meaningful correlation: more items does not mean longer delivery")
    elif abs(pearson[0]) >= 0.3:
        print("-> meaningful correlation")
    else:
        print("-> statistically significant but weak correlation")

    plt.scatter(d["n_items"], d["delivery_days"], alpha=0.4)
    plt.xlabel("Items in order (total quantity)"); plt.ylabel("Delivery days")
    plt.title(f"Items per order vs delivery time (r={pearson[0]:.2f})")
    show("4c_scatter")


def customers_without_orders(data: Datasets) -> None:
    never = ~data.customers["id"].isin(data.orders["customer_id"])
    print(f"Customers who never created an order: {never.sum()} of {len(data.customers)}")


def least_ordered_items(data: Datasets, n: int = 5) -> None:
    qty = data.order_item.groupby("item_id")["quantity"].sum().rename("total_ordered")
    bottom = qty.nsmallest(n).reset_index().merge(data.items[["item_id", "item_name", "item_brand"]], on="item_id")
    print(bottom)
    never_ordered = (~data.items["item_id"].isin(qty.index)).sum()
    print(f"(Items never ordered at all: {never_ordered}, excluded above)")


def analyze_combined(data: Datasets) -> Datasets:
    heading("4. Combined analysis")
    data = add_order_totals(data)
    order_total_stats(data.orders)
    items_vs_delivery(data.orders)
    customers_without_orders(data)
    least_ordered_items(data)
    return data


def usa_campaign(data: Datasets) -> None:
    heading("Bonus 1: USA website campaign")
    orders = data.orders.dropna(subset=["order_date"]).merge(data.customers[["id", "nationality"]], left_on="customer_id", right_on="id", how="left")
    orders["month"] = orders["order_date"].dt.month
    online = orders[orders["order_source"].str.lower() == "online"]

    by_address = online["delivery_address"].astype(str).str.contains(USA_PATTERN, case=False, regex=True)
    by_nationality = online["nationality"].astype(str).str.contains(USA_PATTERN, case=False, regex=True)
    is_usa, basis = (by_address, "delivery address") if by_address.any() else (by_nationality, "nationality")
    online = online.assign(is_usa=is_usa)
    print(f"USA identified by {basis}: {is_usa.sum()} online USA orders (check matches in your data!)")

    monthly = (online.groupby(["is_usa", "month"]).agg(orders=("order_id", "count"), revenue=("total_price", "sum")).unstack(0).reindex(range(1, 13)).fillna(0))

    rows = []
    for metric in ("orders", "revenue"):
        for flag, label in ((True, "USA"), (False, "Rest of world")):
            s = monthly[(metric, flag)]
            pre, post = s.loc[1:CAMPAIGN_MONTH - 1].mean(), s.loc[CAMPAIGN_MONTH:].mean()
            rows.append([metric, label, pre, post, (post / pre - 1) * 100 if pre else np.nan])
    summary = pd.DataFrame(rows, columns=["metric", "group", "avg before", "avg after", "change %"])
    print(summary.round(1))
    for metric in ("orders", "revenue"):
        chg = summary[summary["metric"] == metric].set_index("group")["change %"]
        print(f"{metric}: USA {chg['USA']:.1f}% vs rest of world {chg['Rest of world']:.1f}% -> difference-in-differences {chg['USA'] - chg['Rest of world']:.1f} pts")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ("orders", "revenue")):
        ax.plot(monthly.index, monthly[(metric, True)], marker="o", label="USA")
        ax.plot(monthly.index, monthly[(metric, False)], marker="o", label="Rest of world")
        ax.axvline(CAMPAIGN_MONTH - 0.5, color="red", ls="--", label="campaign start")
        ax.set_title(f"Online {metric} per month (2023)"); ax.set_xlabel("Month"); ax.legend()
    show("bonus1_campaign")


def gift_card_discount(data: Datasets) -> None:
    heading("Bonus 2: August gift-card discount")
    orders = data.orders.dropna(subset=["order_date"]).copy()
    orders["month"] = orders["order_date"].dt.month
    orders["gift"] = orders["payment_method"].astype(str).str.contains("gift", case=False)
    print("Payment methods seen:", sorted(orders["payment_method"].dropna().unique()))

    overall = orders.groupby("month").agg(orders=("order_id", "count"), revenue=("total_price", "sum"))
    gift = orders[orders["gift"]].groupby("month").agg(gift_orders=("order_id", "count"), gift_revenue=("total_price", "sum"))
    table = overall.join(gift).reindex(range(1, 13)).fillna(0)
    table["gift_share_%"] = table["gift_orders"] / table["orders"] * 100
    print(table.round(1))

    aug, rest = table.loc[CAMPAIGN_MONTH], table.drop(CAMPAIGN_MONTH)
    for col in ("orders", "revenue", "gift_orders", "gift_revenue", "gift_share_%"):
        std = rest[col].std(ddof=1)
        z = (aug[col] - rest[col].mean()) / std if std else np.nan
        print(f"{col:13s} Aug={aug[col]:10.1f}  others avg={rest[col].mean():10.1f} diff={(aug[col] / rest[col].mean() - 1) * 100:6.1f}%  z={z:.2f}")

    colors = ["tab:red" if m == CAMPAIGN_MONTH else "steelblue" for m in table.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(table.index, table["revenue"], color=colors); axes[0].set_title("Revenue per month (Aug in red)")
    axes[1].bar(table.index, table["gift_orders"], color=colors); axes[1].set_title("Gift-card orders per month (Aug in red)")
    for ax in axes:
        ax.set_xlabel("Month")
    show("bonus2_giftcard")
    print("\nRule of thumb: August is an increase only if clearly above the other months (|z| > ~1.5) for gift-card orders AND revenue; a discount lowers revenue per order, so read both.")


def main(data_dir: str = ".") -> None:
    raw = load_data(Path(data_dir))
    data = prepare_data(raw)
    analyze_customers(data)
    analyze_items(data)
    analyze_orders(data)
    data = analyze_combined(data)
    usa_campaign(data)
    gift_card_discount(data)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")