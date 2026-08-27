from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/error_analysis")

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
print("STEP 5 - ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["week_of_year"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["weekday_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["weekday_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    data["route"] = (
        data["pickup"].astype(str)
        + " → "
        + data["delivery"].astype(str)
    )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    data["distance_log"] = np.log1p(
        data["distance"].clip(lower=0)
    )

    data["distance_squared"] = (
        data["distance"] ** 2
    )

    # --------------------------------------------------------
    # WEIGHT
    # --------------------------------------------------------

    data["weight_log"] = np.log1p(
        data["weight"].clip(lower=0)
    )

    data["weight_squared"] = (
        data["weight"] ** 2
    )

    # --------------------------------------------------------
    # DISTANCE / WEIGHT
    # --------------------------------------------------------

    safe_distance = data["distance"].replace(
        0,
        np.nan
    )

    data["weight_per_mile"] = (
        data["weight"] /
        safe_distance
    )

    data["distance_weight_interaction"] = (
        data["distance"] *
        data["weight"]
    )

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    data["market_index_squared"] = (
        data["market_index"] ** 2
    )

    data["market_index_log"] = np.log1p(
        data["market_index"].clip(lower=0)
    )

    # --------------------------------------------------------
    # QUOTE SIGNAL
    # --------------------------------------------------------

    data["quote_signal_squared"] = (
        data["quote_signal"] ** 2
    )

    data["quote_signal_abs"] = (
        data["quote_signal"].abs()
    )

    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    data["lat_difference"] = (
        data["delivery_lat"]
        - data["pickup_lat"]
    ).abs()

    data["lon_difference"] = (
        data["delivery_lon"]
        - data["pickup_lon"]
    ).abs()

    data["geo_distance"] = np.sqrt(
        (
            data["delivery_lat"]
            - data["pickup_lat"]
        ) ** 2
        +
        (
            data["delivery_lon"]
            - data["pickup_lon"]
        ) ** 2
    )

    return data


df = create_features(df)


# ============================================================
# TIME-BASED SPLIT
# ============================================================

split_date = pd.Timestamp(
    "2025-09-01"
)

train_df = df[
    df["date"] < split_date
].copy()

valid_df = df[
    df["date"] >= split_date
].copy()


# ============================================================
# PREPARE X / Y
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "load_id",
    "date"
]

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]

X_valid = valid_df.drop(
    columns=DROP_COLUMNS
)

y_valid = valid_df[TARGET]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

predictions = model.predict(
    X_valid
)


# ============================================================
# CREATE ERROR DATAFRAME
# ============================================================

results = valid_df[
    [
        "load_id",
        "date",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        "weight",
        TARGET
    ]
].copy()

results["predicted_rate"] = predictions

results["error"] = (
    results["predicted_rate"]
    - results[TARGET]
)

results["absolute_error"] = (
    results["error"]
    .abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    / results[TARGET].replace(0, np.nan)
    * 100
)


# ============================================================
# OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL VALIDATION PERFORMANCE")
print("=" * 70)

mae = mean_absolute_error(
    y_valid,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        predictions
    )
)

r2 = r2_score(
    y_valid,
    predictions
)

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 1. BIGGEST ERRORS
# ============================================================

print("\n" + "=" * 70)
print("1. TOP 20 LARGEST ERRORS")
print("=" * 70)

largest_errors = results.sort_values(
    "absolute_error",
    ascending=False
).head(20)

print(
    largest_errors[
        [
            "date",
            "pickup",
            "delivery",
            "equipment",
            "distance",
            TARGET,
            "predicted_rate",
            "absolute_error"
        ]
    ].to_string(index=False)
)


# ============================================================
# 2. ERROR BY EQUIPMENT
# ============================================================

print("\n" + "=" * 70)
print("2. ERROR BY EQUIPMENT")
print("=" * 70)

equipment_error = (
    results.groupby("equipment")
    .agg(
        count=("absolute_error", "count"),
        MAE=("absolute_error", "mean"),
        median_error=("absolute_error", "median"),
        mean_actual=(TARGET, "mean"),
        mean_predicted=("predicted_rate", "mean")
    )
    .sort_values("MAE", ascending=False)
)

print(
    equipment_error.to_string()
)


# ============================================================
# 3. ERROR BY DISTANCE
# ============================================================

results["distance_bucket"] = pd.cut(
    results["distance"],
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
    ]
)

distance_error = (
    results.groupby(
        "distance_bucket",
        observed=True
    )
    .agg(
        count=("absolute_error", "count"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "error",
            lambda x: np.sqrt(
                np.mean(x ** 2)
            )
        )
    )
)

print("\n" + "=" * 70)
print("3. ERROR BY DISTANCE")
print("=" * 70)

print(
    distance_error.to_string()
)


# ============================================================
# 4. ERROR BY MONTH
# ============================================================

results["month"] = (
    results["date"].dt.month
)

monthly_error = (
    results.groupby("month")
    .agg(
        count=("absolute_error", "count"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "error",
            lambda x: np.sqrt(
                np.mean(x ** 2)
            )
        )
    )
)

print("\n" + "=" * 70)
print("4. ERROR BY MONTH")
print("=" * 70)

print(
    monthly_error.to_string()
)


# ============================================================
# 5. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    results[TARGET],
    results["predicted_rate"],
    alpha=0.25,
    s=12
)

min_value = min(
    results[TARGET].min(),
    results["predicted_rate"].min()
)

max_value = max(
    results[TARGET].max(),
    results["predicted_rate"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Freight Rate"
)

plt.xlabel(
    "Actual Posted Rate ($)"
)

plt.ylabel(
    "Predicted Rate ($)"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "01_actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# 6. RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    results["error"],
    bins=60
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title(
    "Prediction Error Distribution"
)

plt.xlabel(
    "Prediction Error ($)"
)

plt.ylabel(
    "Number of Loads"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "02_error_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 7. ERROR VS DISTANCE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    results["distance"],
    results["absolute_error"],
    alpha=0.25,
    s=12
)

plt.title(
    "Absolute Prediction Error vs Distance"
)

plt.xlabel(
    "Distance (miles)"
)

plt.ylabel(
    "Absolute Error ($)"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "03_error_vs_distance.png",
    dpi=150
)

plt.close()


# ============================================================
# 8. ACTUAL RATE VS ERROR
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    results[TARGET],
    results["absolute_error"],
    alpha=0.25,
    s=12
)

plt.title(
    "Absolute Error vs Actual Rate"
)

plt.xlabel(
    "Actual Posted Rate ($)"
)

plt.ylabel(
    "Absolute Error ($)"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "04_error_vs_actual_rate.png",
    dpi=150
)

plt.close()


# ============================================================
# 9. SAVE ERROR DATA
# ============================================================

results.to_csv(
    OUTPUT_DIR /
    "validation_error_analysis.csv",
    index=False
)

equipment_error.to_csv(
    OUTPUT_DIR /
    "equipment_error_summary.csv"
)

distance_error.to_csv(
    OUTPUT_DIR /
    "distance_error_summary.csv"
)

monthly_error.to_csv(
    OUTPUT_DIR /
    "monthly_error_summary.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFiles saved to:"
)

print(
    OUTPUT_DIR.resolve()
)