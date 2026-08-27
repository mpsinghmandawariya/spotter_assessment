from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 6 - GRADIENT BOOSTING MODELS")
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
    # DATE FEATURES
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

    # Cyclic features

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
    # MARKET INDEX
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
    SPLIT_DATE
)

train_df = df[
    df["date"] < split_date
].copy()

valid_df = df[
    df["date"] >= split_date
].copy()


print("\nTraining rows:", len(train_df))
print("Validation rows:", len(valid_df))

print(
    "\nTraining period:",
    train_df["date"].min().date(),
    "→",
    train_df["date"].max().date()
)

print(
    "Validation period:",
    valid_df["date"].min().date(),
    "→",
    valid_df["date"].max().date()
)


# ============================================================
# PREPARE FEATURES
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
# MODEL FACTORY
# ============================================================

def build_model():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=700,
                    learning_rate=0.05,
                    max_depth=8,
                    min_child_weight=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(
        f"MAE  : ${mae:,.2f}"
    )

    print(
        f"RMSE : ${rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# MODEL 1 — RAW TARGET
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 - XGBOOST RAW TARGET")
print("=" * 70)

model_raw = build_model()

print("\nTraining...")

model_raw.fit(
    X_train,
    y_train
)

raw_predictions = model_raw.predict(
    X_valid
)

raw_results = evaluate(
    "XGBoost - Raw Target",
    y_valid,
    raw_predictions
)


# ============================================================
# MODEL 2 — LOG TARGET
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 - XGBOOST LOG TARGET")
print("=" * 70)

model_log = build_model()

log_y_train = np.log1p(
    y_train
)

print("\nTraining...")

model_log.fit(
    X_train,
    log_y_train
)

log_predictions = model_log.predict(
    X_valid
)

# Convert back to dollars

log_predictions = np.expm1(
    log_predictions
)

# Prevent negative predictions

log_predictions = np.maximum(
    log_predictions,
    0
)

log_results = evaluate(
    "XGBoost - Log Target",
    y_valid,
    log_predictions
)


# ============================================================
# COMPARE
# ============================================================

results = pd.DataFrame([
    raw_results,
    log_results
])

print("\n" + "=" * 70)
print("XGBOOST COMPARISON")
print("=" * 70)

print(
    results
    .sort_values("RMSE")
    .to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_DIR /
    "xgboost_results.csv",
    index=False
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

best_predictions = (
    raw_predictions
    if raw_results["RMSE"] <= log_results["RMSE"]
    else log_predictions
)

error_df = valid_df[
    [
        "date",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        TARGET
    ]
].copy()

error_df["predicted_rate"] = (
    best_predictions
)

error_df["absolute_error"] = (
    error_df["predicted_rate"]
    - error_df[TARGET]
).abs()

error_df["percentage_error"] = (
    error_df["absolute_error"]
    /
    error_df[TARGET]
    *
    100
)


print("\n" + "=" * 70)
print("TOP 20 XGBOOST ERRORS")
print("=" * 70)

print(
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE ERROR DATA
# ============================================================

error_df.to_csv(
    OUTPUT_DIR /
    "xgboost_error_analysis.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:",
    OUTPUT_DIR / "xgboost_results.csv"
)