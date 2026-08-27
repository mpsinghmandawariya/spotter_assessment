from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_test.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FREIGHT RATE PREDICTION - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# DATE CONVERSION
# ============================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")


# ============================================================
# TARGET
# ============================================================

TARGET = "posted_rate"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("1. TARGET DISTRIBUTION")
print("=" * 70)

print(df[TARGET].describe())

plt.figure(figsize=(10, 6))

plt.hist(
    df[TARGET].dropna(),
    bins=50
)

plt.title("Distribution of Posted Freight Rate")
plt.xlabel("Posted Rate ($)")
plt.ylabel("Number of Loads")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_target_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 2. TARGET BOX PLOT
# ============================================================

plt.figure(figsize=(10, 4))

plt.boxplot(
    df[TARGET].dropna(),
    vert=False
)

plt.title("Posted Rate - Box Plot")
plt.xlabel("Posted Rate ($)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_target_boxplot.png",
    dpi=150
)

plt.close()


# ============================================================
# 3. NUMERICAL CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("2. NUMERICAL CORRELATIONS WITH TARGET")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

correlations = (
    df[numeric_columns]
    .corr()[TARGET]
    .sort_values(ascending=False)
)

print(correlations)


# ============================================================
# 4. CORRELATION BAR CHART
# ============================================================

correlations_without_target = correlations.drop(
    TARGET,
    errors="ignore"
)

plt.figure(figsize=(10, 6))

correlations_without_target.sort_values().plot(
    kind="barh"
)

plt.title("Correlation of Numerical Features with Posted Rate")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_feature_correlations.png",
    dpi=150
)

plt.close()


# ============================================================
# 5. DISTANCE VS RATE
# ============================================================

if "distance" in df.columns:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["distance"],
        df[TARGET],
        alpha=0.25,
        s=10
    )

    plt.title("Distance vs Posted Rate")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Posted Rate ($)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "04_distance_vs_rate.png",
        dpi=150
    )

    plt.close()

    print(
        "\nDistance correlation:",
        df["distance"].corr(df[TARGET])
    )


# ============================================================
# 6. WEIGHT VS RATE
# ============================================================

if "weight" in df.columns:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["weight"],
        df[TARGET],
        alpha=0.25,
        s=10
    )

    plt.title("Weight vs Posted Rate")
    plt.xlabel("Weight (lb)")
    plt.ylabel("Posted Rate ($)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "05_weight_vs_rate.png",
        dpi=150
    )

    plt.close()

    print(
        "Weight correlation:",
        df["weight"].corr(df[TARGET])
    )


# ============================================================
# 7. MARKET INDEX VS RATE
# ============================================================

if "market_index" in df.columns:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["market_index"],
        df[TARGET],
        alpha=0.25,
        s=10
    )

    plt.title("Market Index vs Posted Rate")
    plt.xlabel("Market Index")
    plt.ylabel("Posted Rate ($)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "06_market_index_vs_rate.png",
        dpi=150
    )

    plt.close()

    print(
        "Market index correlation:",
        df["market_index"].corr(df[TARGET])
    )


# ============================================================
# 8. QUOTE SIGNAL VS RATE
# ============================================================

if "quote_signal" in df.columns:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["quote_signal"],
        df[TARGET],
        alpha=0.25,
        s=10
    )

    plt.title("Quote Signal vs Posted Rate")
    plt.xlabel("Quote Signal")
    plt.ylabel("Posted Rate ($)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "07_quote_signal_vs_rate.png",
        dpi=150
    )

    plt.close()

    print(
        "Quote signal correlation:",
        df["quote_signal"].corr(df[TARGET])
    )


# ============================================================
# 9. EQUIPMENT VS RATE
# ============================================================

if "equipment" in df.columns:

    equipment_summary = (
        df.groupby("equipment")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max"
        )
        .sort_values("mean", ascending=False)
    )

    print("\n" + "=" * 70)
    print("3. EQUIPMENT ANALYSIS")
    print("=" * 70)

    print(equipment_summary)

    equipment_summary["mean"].plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Average Posted Rate by Equipment")
    plt.xlabel("Equipment")
    plt.ylabel("Average Posted Rate ($)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "08_equipment_vs_rate.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 10. TOP PICKUP LOCATIONS
# ============================================================

if "pickup" in df.columns:

    pickup_summary = (
        df.groupby("pickup")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median"
        )
        .sort_values("count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("4. TOP PICKUP LOCATIONS")
    print("=" * 70)

    print(pickup_summary.head(20))


# ============================================================
# 11. TOP DELIVERY LOCATIONS
# ============================================================

if "delivery" in df.columns:

    delivery_summary = (
        df.groupby("delivery")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median"
        )
        .sort_values("count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("5. TOP DELIVERY LOCATIONS")
    print("=" * 70)

    print(delivery_summary.head(20))


# ============================================================
# 12. ROUTE ANALYSIS
# ============================================================

if "pickup" in df.columns and "delivery" in df.columns:

    df["route"] = (
        df["pickup"].astype(str)
        + " → "
        + df["delivery"].astype(str)
    )

    route_summary = (
        df.groupby("route")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median"
        )
        .sort_values("count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("6. MOST FREQUENT ROUTES")
    print("=" * 70)

    print(route_summary.head(20))

    top_routes = route_summary.head(15)

    plt.figure(figsize=(12, 7))

    top_routes["mean"].sort_values().plot(
        kind="barh"
    )

    plt.title("Average Posted Rate - Top Routes by Frequency")
    plt.xlabel("Average Posted Rate ($)")
    plt.ylabel("Route")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "09_route_rates.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 13. DAILY RATE TREND
# ============================================================

if "date" in df.columns:

    daily_rate = (
        df.groupby("date")[TARGET]
        .agg(
            mean="mean",
            median="median",
            count="count"
        )
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("7. DAILY RATE TREND")
    print("=" * 70)

    print(daily_rate.head())
    print("\n...")
    print(daily_rate.tail())

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily_rate.index,
        daily_rate["mean"],
        marker="o",
        markersize=2
    )

    plt.title("Daily Average Posted Rate")
    plt.xlabel("Date")
    plt.ylabel("Average Posted Rate ($)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "10_daily_rate_trend.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 14. MONTHLY RATE TREND
# ============================================================

if "date" in df.columns:

    df["month"] = df["date"].dt.to_period("M")

    monthly_rate = (
        df.groupby("month")[TARGET]
        .agg(
            mean="mean",
            median="median",
            count="count"
        )
    )

    print("\n" + "=" * 70)
    print("8. MONTHLY RATE TREND")
    print("=" * 70)

    print(monthly_rate)

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_rate.index.astype(str),
        monthly_rate["mean"],
        marker="o"
    )

    plt.title("Monthly Average Posted Rate")
    plt.xlabel("Month")
    plt.ylabel("Average Posted Rate ($)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "11_monthly_rate_trend.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 15. RATE PER MILE
# ============================================================

if "distance" in df.columns:

    df["rate_per_mile"] = (
        df[TARGET] /
        df["distance"].replace(0, np.nan)
    )

    print("\n" + "=" * 70)
    print("9. RATE PER MILE")
    print("=" * 70)

    print(
        df["rate_per_mile"].describe()
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["rate_per_mile"].dropna(),
        bins=50
    )

    plt.title("Distribution of Rate per Mile")
    plt.xlabel("Rate per Mile ($/mile)")
    plt.ylabel("Number of Loads")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "12_rate_per_mile.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 16. WEEKDAY EFFECT
# ============================================================

if "date" in df.columns:

    df["weekday"] = df["date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_summary = (
        df.groupby("weekday")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median"
        )
        .reindex(weekday_order)
    )

    print("\n" + "=" * 70)
    print("10. WEEKDAY ANALYSIS")
    print("=" * 70)

    print(weekday_summary)

    plt.figure(figsize=(10, 6))

    weekday_summary["mean"].plot(
        kind="bar"
    )

    plt.title("Average Posted Rate by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Posted Rate ($)")
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "13_weekday_rates.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 17. SAVE SUMMARY TABLES
# ============================================================

if "date" in df.columns:

    monthly_rate.to_csv(
        OUTPUT_DIR / "monthly_rate_summary.csv"
    )

if "equipment" in df.columns:

    equipment_summary.to_csv(
        OUTPUT_DIR / "equipment_summary.csv"
    )

if "pickup" in df.columns:

    pickup_summary.to_csv(
        OUTPUT_DIR / "pickup_summary.csv"
    )

if "delivery" in df.columns:

    delivery_summary.to_csv(
        OUTPUT_DIR / "delivery_summary.csv"
    )

if "pickup" in df.columns and "delivery" in df.columns:

    route_summary.to_csv(
        OUTPUT_DIR / "route_summary.csv"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"\nCharts and summaries saved to:")
print(OUTPUT_DIR.resolve())

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nNext step: analyze these results before model training.")