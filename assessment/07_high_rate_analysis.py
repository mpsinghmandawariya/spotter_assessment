from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/high_rate_analysis")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 7 - HIGH RATE ANALYSIS")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# CREATE ROUTE
# ============================================================

df["route"] = (
    df["pickup"].astype(str)
    + " → "
    + df["delivery"].astype(str)
)


# ============================================================
# RATE PER MILE
# ============================================================

df["rate_per_mile"] = (
    df[TARGET]
    /
    df["distance"].replace(0, np.nan)
)


# ============================================================
# RATE SEGMENTS
# ============================================================

df["rate_segment"] = pd.cut(
    df[TARGET],
    bins=[
        -np.inf,
        1000,
        2000,
        3000,
        5000,
        7500,
        10000,
        15000,
        np.inf
    ],
    labels=[
        "<1000",
        "1000-2000",
        "2000-3000",
        "3000-5000",
        "5000-7500",
        "7500-10000",
        "10000-15000",
        "15000+"
    ]
)


# ============================================================
# BASIC SEGMENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("1. RATE SEGMENTS")
print("=" * 70)

segment_summary = (
    df.groupby(
        "rate_segment",
        observed=True
    )
    .agg(
        count=(TARGET, "count"),
        mean_rate=(TARGET, "mean"),
        median_rate=(TARGET, "median"),
        mean_distance=("distance", "mean"),
        mean_weight=("weight", "mean"),
        mean_market_index=("market_index", "mean"),
        mean_quote_signal=("quote_signal", "mean")
    )
)

segment_summary["percentage"] = (
    segment_summary["count"]
    /
    len(df)
    *
    100
)

print(
    segment_summary.to_string()
)


# ============================================================
# EXTREME RATE DATASET
# ============================================================

HIGH_RATE_THRESHOLD = 7500

high_rate = df[
    df[TARGET] >= HIGH_RATE_THRESHOLD
].copy()

normal_rate = df[
    df[TARGET] < HIGH_RATE_THRESHOLD
].copy()


print("\n" + "=" * 70)
print("2. HIGH RATE LOADS")
print("=" * 70)

print(
    f"Threshold: ${HIGH_RATE_THRESHOLD:,}"
)

print(
    f"High-rate loads: {len(high_rate):,}"
)

print(
    f"Percentage of dataset: "
    f"{len(high_rate) / len(df) * 100:.3f}%"
)


# ============================================================
# HIGH RATE FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("3. HIGH RATE VS NORMAL RATE")
print("=" * 70)

comparison = pd.DataFrame({
    "Normal": normal_rate[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "rate_per_mile"
        ]
    ].mean(),

    "High Rate": high_rate[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "rate_per_mile"
        ]
    ].mean()
})

comparison["difference_%"] = (
    (
        comparison["High Rate"]
        -
        comparison["Normal"]
    )
    /
    comparison["Normal"].replace(0, np.nan)
    *
    100
)

print(
    comparison.to_string()
)


# ============================================================
# 4. HIGH RATE BY EQUIPMENT
# ============================================================

print("\n" + "=" * 70)
print("4. HIGH RATE BY EQUIPMENT")
print("=" * 70)

equipment_high = (
    df.groupby("equipment")
    .agg(
        total_loads=(TARGET, "count"),
        high_rate_loads=(
            TARGET,
            lambda x: (x >= HIGH_RATE_THRESHOLD).sum()
        ),
        mean_rate=(TARGET, "mean"),
        max_rate=(TARGET, "max")
    )
)

equipment_high["high_rate_percentage"] = (
    equipment_high["high_rate_loads"]
    /
    equipment_high["total_loads"]
    *
    100
)

print(
    equipment_high.to_string()
)


# ============================================================
# 5. HIGH RATE BY MONTH
# ============================================================

print("\n" + "=" * 70)
print("5. HIGH RATE BY MONTH")
print("=" * 70)

high_rate["month"] = (
    high_rate["date"].dt.month
)

monthly_high = (
    df.assign(
        month=df["date"].dt.month
    )
    .groupby("month")
    .agg(
        total_loads=(TARGET, "count"),
        high_rate_loads=(
            TARGET,
            lambda x: (x >= HIGH_RATE_THRESHOLD).sum()
        ),
        mean_rate=(TARGET, "mean")
    )
)

monthly_high["high_rate_percentage"] = (
    monthly_high["high_rate_loads"]
    /
    monthly_high["total_loads"]
    *
    100
)

print(
    monthly_high.to_string()
)


# ============================================================
# 6. HIGH RATE BY DISTANCE
# ============================================================

print("\n" + "=" * 70)
print("6. HIGH RATE BY DISTANCE")
print("=" * 70)

df["distance_bucket"] = pd.cut(
    df["distance"],
    bins=[
        -np.inf,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        np.inf
    ],
    labels=[
        "<500",
        "500-1000",
        "1000-1500",
        "1500-2000",
        "2000-2500",
        "2500-3000",
        "3000+"
    ]
)

distance_high = (
    df.groupby(
        "distance_bucket",
        observed=True
    )
    .agg(
        total_loads=(TARGET, "count"),
        high_rate_loads=(
            TARGET,
            lambda x: (x >= HIGH_RATE_THRESHOLD).sum()
        ),
        mean_rate=(TARGET, "mean"),
        median_rate=(TARGET, "median")
    )
)

distance_high["high_rate_percentage"] = (
    distance_high["high_rate_loads"]
    /
    distance_high["total_loads"]
    *
    100
)

print(
    distance_high.to_string()
)


# ============================================================
# 7. TOP HIGH-RATE ROUTES
# ============================================================

print("\n" + "=" * 70)
print("7. ROUTES WITH HIGH-RATE LOADS")
print("=" * 70)

route_high = (
    df.groupby("route")
    .agg(
        total_loads=(TARGET, "count"),
        high_rate_loads=(
            TARGET,
            lambda x: (x >= HIGH_RATE_THRESHOLD).sum()
        ),
        mean_rate=(TARGET, "mean"),
        max_rate=(TARGET, "max")
    )
)

route_high = route_high[
    route_high["high_rate_loads"] > 0
].copy()

route_high["high_rate_percentage"] = (
    route_high["high_rate_loads"]
    /
    route_high["total_loads"]
    *
    100
)

route_high = route_high.sort_values(
    "high_rate_loads",
    ascending=False
)

print(
    route_high.head(30).to_string()
)


# ============================================================
# 8. TOP HIGH-RATE PICKUP LOCATIONS
# ============================================================

print("\n" + "=" * 70)
print("8. HIGH-RATE PICKUP LOCATIONS")
print("=" * 70)

pickup_high = (
    high_rate.groupby("pickup")
    .agg(
        high_rate_loads=(TARGET, "count"),
        mean_rate=(TARGET, "mean"),
        max_rate=(TARGET, "max")
    )
    .sort_values(
        "high_rate_loads",
        ascending=False
    )
)

print(
    pickup_high.head(30).to_string()
)


# ============================================================
# 9. TOP HIGH-RATE DELIVERY LOCATIONS
# ============================================================

print("\n" + "=" * 70)
print("9. HIGH-RATE DELIVERY LOCATIONS")
print("=" * 70)

delivery_high = (
    high_rate.groupby("delivery")
    .agg(
        high_rate_loads=(TARGET, "count"),
        mean_rate=(TARGET, "mean"),
        max_rate=(TARGET, "max")
    )
    .sort_values(
        "high_rate_loads",
        ascending=False
    )
)

print(
    delivery_high.head(30).to_string()
)


# ============================================================
# 10. CORRELATIONS INSIDE HIGH-RATE LOADS
# ============================================================

print("\n" + "=" * 70)
print("10. HIGH-RATE CORRELATIONS")
print("=" * 70)

numeric_columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "rate_per_mile",
    TARGET
]

high_correlations = (
    high_rate[numeric_columns]
    .corr()[TARGET]
    .sort_values(
        ascending=False
    )
)

print(
    high_correlations.to_string()
)


# ============================================================
# 11. EXTREME LOADS
# ============================================================

print("\n" + "=" * 70)
print("11. TOP 30 MOST EXPENSIVE LOADS")
print("=" * 70)

top_expensive = (
    df.sort_values(
        TARGET,
        ascending=False
    )
    .head(30)
)

columns_to_show = [
    "date",
    "pickup",
    "delivery",
    "equipment",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    TARGET,
    "rate_per_mile"
]

print(
    top_expensive[
        columns_to_show
    ].to_string(index=False)
)


# ============================================================
# 12. QUOTE SIGNAL BINS
# ============================================================

print("\n" + "=" * 70)
print("12. QUOTE SIGNAL ANALYSIS")
print("=" * 70)

df["quote_signal_bucket"] = pd.qcut(
    df["quote_signal"],
    q=10,
    duplicates="drop"
)

quote_summary = (
    df.groupby(
        "quote_signal_bucket",
        observed=True
    )
    .agg(
        count=(TARGET, "count"),
        mean_rate=(TARGET, "mean"),
        median_rate=(TARGET, "median"),
        max_rate=(TARGET, "max")
    )
)

print(
    quote_summary.to_string()
)


# ============================================================
# 13. MARKET INDEX BINS
# ============================================================

print("\n" + "=" * 70)
print("13. MARKET INDEX ANALYSIS")
print("=" * 70)

df["market_index_bucket"] = pd.qcut(
    df["market_index"],
    q=10,
    duplicates="drop"
)

market_summary = (
    df.groupby(
        "market_index_bucket",
        observed=True
    )
    .agg(
        count=(TARGET, "count"),
        mean_rate=(TARGET, "mean"),
        median_rate=(TARGET, "median"),
        max_rate=(TARGET, "max")
    )
)

print(
    market_summary.to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

segment_summary.to_csv(
    OUTPUT_DIR /
    "rate_segments.csv"
)

comparison.to_csv(
    OUTPUT_DIR /
    "high_vs_normal.csv"
)

equipment_high.to_csv(
    OUTPUT_DIR /
    "high_rate_equipment.csv"
)

monthly_high.to_csv(
    OUTPUT_DIR /
    "high_rate_monthly.csv"
)

distance_high.to_csv(
    OUTPUT_DIR /
    "high_rate_distance.csv"
)

route_high.to_csv(
    OUTPUT_DIR /
    "high_rate_routes.csv"
)

pickup_high.to_csv(
    OUTPUT_DIR /
    "high_rate_pickups.csv"
)

delivery_high.to_csv(
    OUTPUT_DIR /
    "high_rate_deliveries.csv"
)

quote_summary.to_csv(
    OUTPUT_DIR /
    "quote_signal_analysis.csv"
)

market_summary.to_csv(
    OUTPUT_DIR /
    "market_index_analysis.csv"
)


print("\n" + "=" * 70)
print("HIGH RATE ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR.resolve()
)