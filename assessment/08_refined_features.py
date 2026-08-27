from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/refined_features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"


# ============================================================
# LOAD
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)


# ============================================================
# FEATURE FUNCTION
# ============================================================

def create_refined_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # CYCLIC DATE FEATURES
    # --------------------------------------------------------

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["weekday_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    df["route"] = (
        df["pickup"].astype(str)
        + " → "
        + df["delivery"].astype(str)
    )

    # Route + equipment interaction

    df["route_equipment"] = (
        df["route"].astype(str)
        + " | "
        + df["equipment"].astype(str)
    )

    # --------------------------------------------------------
    # DISTANCE FEATURES
    # --------------------------------------------------------

    distance = pd.to_numeric(
        df["distance"],
        errors="coerce"
    )

    df["distance_log"] = np.log1p(
        distance.clip(lower=0)
    )

    df["distance_squared"] = (
        distance ** 2
    )

    df["distance_cubed"] = (
        distance ** 3
    )

    # Long-haul indicators

    df["long_haul_500"] = (
        distance >= 500
    ).astype(int)

    df["long_haul_1000"] = (
        distance >= 1000
    ).astype(int)

    df["long_haul_1500"] = (
        distance >= 1500
    ).astype(int)

    df["long_haul_2000"] = (
        distance >= 2000
    ).astype(int)

    df["long_haul_2500"] = (
        distance >= 2500
    ).astype(int)

    # Distance buckets

    df["distance_bucket"] = pd.cut(
        distance,
        bins=[
            -np.inf,
            100,
            250,
            500,
            750,
            1000,
            1500,
            2000,
            2500,
            3000,
            np.inf
        ],
        labels=[
            "0-100",
            "100-250",
            "250-500",
            "500-750",
            "750-1000",
            "1000-1500",
            "1500-2000",
            "2000-2500",
            "2500-3000",
            "3000+"
        ]
    )

    # --------------------------------------------------------
    # WEIGHT
    # --------------------------------------------------------

    weight = pd.to_numeric(
        df["weight"],
        errors="coerce"
    )

    df["weight_log"] = np.log1p(
        weight.clip(lower=0)
    )

    df["weight_squared"] = (
        weight ** 2
    )

    # Weight categories

    df["heavy_load"] = (
        weight >= 35000
    ).astype(int)

    df["very_heavy_load"] = (
        weight >= 40000
    ).astype(int)

    # --------------------------------------------------------
    # DISTANCE × WEIGHT
    # --------------------------------------------------------

    df["distance_weight"] = (
        distance * weight
    )

    safe_distance = distance.replace(
        0,
        np.nan
    )

    df["weight_per_mile"] = (
        weight / safe_distance
    )

    # --------------------------------------------------------
    # MARKET INDEX
    # --------------------------------------------------------

    market = pd.to_numeric(
        df["market_index"],
        errors="coerce"
    )

    df["market_index_squared"] = (
        market ** 2
    )

    df["market_index_centered"] = (
        market - 1.0
    )

    df["market_index_high"] = (
        market >= 1.2
    ).astype(int)

    df["market_index_very_high"] = (
        market >= 1.3
    ).astype(int)

    # --------------------------------------------------------
    # QUOTE SIGNAL
    # --------------------------------------------------------

    signal = pd.to_numeric(
        df["quote_signal"],
        errors="coerce"
    )

    # Polynomial features

    df["quote_signal_squared"] = (
        signal ** 2
    )

    df["quote_signal_cubed"] = (
        signal ** 3
    )

    # Distance × quote signal

    df["distance_quote_signal"] = (
        distance * signal
    )

    df["distance_quote_signal_squared"] = (
        distance * signal ** 2
    )

    # Important nonlinear regions discovered in EDA

    df["quote_signal_low"] = (
        signal < 1.735
    ).astype(int)

    df["quote_signal_high"] = (
        signal > 2.403
    ).astype(int)

    df["quote_signal_mid"] = (
        (
            signal >= 1.735
        )
        &
        (
            signal <= 2.403
        )
    ).astype(int)

    # --------------------------------------------------------
    # MARKET × DISTANCE
    # --------------------------------------------------------

    df["distance_market"] = (
        distance * market
    )

    df["distance_market_squared"] = (
        distance * market ** 2
    )

    # --------------------------------------------------------
    # THREE-WAY INTERACTIONS
    # --------------------------------------------------------

    df["distance_market_quote"] = (
        distance
        * market
        * signal
    )

    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    df["lat_difference"] = (
        df["delivery_lat"]
        - df["pickup_lat"]
    ).abs()

    df["lon_difference"] = (
        df["delivery_lon"]
        - df["pickup_lon"]
    ).abs()

    df["geo_distance"] = np.sqrt(
        (
            df["delivery_lat"]
            - df["pickup_lat"]
        ) ** 2
        +
        (
            df["delivery_lon"]
            - df["pickup_lon"]
        ) ** 2
    )

    # --------------------------------------------------------
    # GEOGRAPHY × DISTANCE
    # --------------------------------------------------------

    df["geo_distance_per_mile"] = (
        df["geo_distance"]
        /
        safe_distance
    )

    return df


# ============================================================
# CREATE
# ============================================================

print("=" * 70)
print("CREATING REFINED FEATURES")
print("=" * 70)

train_refined = create_refined_features(
    train
)

validation_refined = create_refined_features(
    validation
)


# ============================================================
# REMOVE TARGET FROM VALIDATION ONLY
# ============================================================

print("\nTraining shape:")
print(train_refined.shape)

print("\nValidation shape:")
print(validation_refined.shape)


# ============================================================
# CHECK LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("TARGET LEAKAGE CHECK")
print("=" * 70)

leakage_keywords = [
    "posted_rate",
    "rate_per_mile"
]

for column in train_refined.columns:

    for keyword in leakage_keywords:

        if keyword in column.lower():

            print(
                "CHECK:",
                column
            )


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = train_refined.isna().sum()

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


# ============================================================
# SAVE
# ============================================================

train_output = (
    OUTPUT_DIR /
    "train_refined.csv"
)

validation_output = (
    OUTPUT_DIR /
    "validation_refined.csv"
)

train_refined.to_csv(
    train_output,
    index=False
)

validation_refined.to_csv(
    validation_output,
    index=False
)


print("\n" + "=" * 70)
print("REFINED FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    "\nTraining saved to:",
    train_output
)

print(
    "Validation saved to:",
    validation_output
)