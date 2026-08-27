from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_year"] = df["date"].dt.dayofyear

        # Weekend indicator
        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        # Cyclic representation of month
        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["month_cos"] = np.cos(
            2 * np.pi * df["month"] / 12
        )

        # Cyclic representation of weekday
        df["weekday_sin"] = np.sin(
            2 * np.pi * df["day_of_week"] / 7
        )

        df["weekday_cos"] = np.cos(
            2 * np.pi * df["day_of_week"] / 7
        )


    # --------------------------------------------------------
    # ROUTE FEATURES
    # --------------------------------------------------------

    if "pickup" in df.columns and "delivery" in df.columns:

        df["route"] = (
            df["pickup"].astype(str)
            + " → "
            + df["delivery"].astype(str)
        )

        df["same_city"] = (
            df["pickup"] == df["delivery"]
        ).astype(int)


    # --------------------------------------------------------
    # DISTANCE FEATURES
    # --------------------------------------------------------

    if "distance" in df.columns:

        distance = pd.to_numeric(
            df["distance"],
            errors="coerce"
        )

        # Avoid division by zero
        safe_distance = distance.replace(
            0,
            np.nan
        )

        df["distance_log"] = np.log1p(
            distance.clip(lower=0)
        )

        df["distance_squared"] = (
            distance ** 2
        )

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
                2500,
                np.inf
            ],
            labels=[
                "0-100",
                "100-250",
                "250-500",
                "500-750",
                "750-1000",
                "1000-1500",
                "1500-2500",
                "2500+"
            ]
        )


    # --------------------------------------------------------
    # WEIGHT FEATURES
    # --------------------------------------------------------

    if "weight" in df.columns:

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


    # --------------------------------------------------------
    # WEIGHT / DISTANCE INTERACTION
    # --------------------------------------------------------

    if (
        "weight" in df.columns
        and "distance" in df.columns
    ):

        safe_distance = (
            df["distance"]
            .replace(0, np.nan)
        )

        df["weight_per_mile"] = (
            df["weight"] /
            safe_distance
        )

        df["distance_weight_interaction"] = (
            df["distance"] *
            df["weight"]
        )


    # --------------------------------------------------------
    # MARKET FEATURES
    # --------------------------------------------------------

    if "market_index" in df.columns:

        market = pd.to_numeric(
            df["market_index"],
            errors="coerce"
        )

        df["market_index_squared"] = (
            market ** 2
        )

        df["market_index_log"] = np.log1p(
            market.clip(lower=0)
        )


    # --------------------------------------------------------
    # QUOTE SIGNAL FEATURES
    # --------------------------------------------------------

    if "quote_signal" in df.columns:

        signal = pd.to_numeric(
            df["quote_signal"],
            errors="coerce"
        )

        df["quote_signal_squared"] = (
            signal ** 2
        )

        df["quote_signal_abs"] = (
            signal.abs()
        )


    # --------------------------------------------------------
    # GEOGRAPHIC FEATURES
    # --------------------------------------------------------

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon"
    ]

    if all(
        col in df.columns
        for col in coordinate_columns
    ):

        # Absolute latitude difference
        df["lat_difference"] = (
            df["delivery_lat"] -
            df["pickup_lat"]
        ).abs()

        # Absolute longitude difference
        df["lon_difference"] = (
            df["delivery_lon"] -
            df["pickup_lon"]
        ).abs()

        # Euclidean coordinate distance
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


    return df


# ============================================================
# CREATE FEATURES
# ============================================================

print("=" * 70)
print("CREATING FEATURES")
print("=" * 70)

train_features = create_features(train)
validation_features = create_features(validation)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\nTraining feature shape:")
print(train_features.shape)

print("\nValidation feature shape:")
print(validation_features.shape)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER FEATURE ENGINEERING")
print("=" * 70)

missing = train_features.isna().sum()

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
# CHECK INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("INFINITE VALUES")
print("=" * 70)

numeric_features = train_features.select_dtypes(
    include=np.number
)

infinite_counts = np.isinf(
    numeric_features
).sum()

infinite_counts = infinite_counts[
    infinite_counts > 0
]

if len(infinite_counts) == 0:
    print("No infinite values.")
else:
    print(infinite_counts)


# ============================================================
# FEATURE LIST
# ============================================================

print("\n" + "=" * 70)
print("FEATURE LIST")
print("=" * 70)

for index, column in enumerate(
    train_features.columns,
    start=1
):
    print(
        f"{index:3}. {column}"
    )


# ============================================================
# SAVE FEATURE DATA
# ============================================================

train_output = OUTPUT_DIR / "train_features.csv"
validation_output = OUTPUT_DIR / "validation_features.csv"

train_features.to_csv(
    train_output,
    index=False
)

validation_features.to_csv(
    validation_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"\nTraining features saved to:\n"
    f"{train_output}"
)

print(
    f"\nValidation features saved to:\n"
    f"{validation_output}"
)