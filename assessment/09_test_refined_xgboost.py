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
OUTPUT_DIR = Path("outputs/refined_features")

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"


# ============================================================
# LOAD ORIGINAL LABELED DATA
# ============================================================

print("=" * 70)
print("STEP 8B - TEST REFINED XGBOOST")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_refined_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

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
    # CYCLIC DATE
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

    df["route_equipment"] = (
        df["route"].astype(str)
        + " | "
        + df["equipment"].astype(str)
    )

    # --------------------------------------------------------
    # DISTANCE
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

    df["quote_signal_squared"] = (
        signal ** 2
    )

    df["quote_signal_cubed"] = (
        signal ** 3
    )

    df["distance_quote_signal"] = (
        distance * signal
    )

    df["distance_quote_signal_squared"] = (
        distance * signal ** 2
    )

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
    # THREE-WAY INTERACTION
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

    df["geo_distance_per_mile"] = (
        df["geo_distance"]
        /
        safe_distance
    )

    return df


# ============================================================
# CREATE FEATURES
# ============================================================

df = create_refined_features(df)


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


print("\nTraining rows:")
print(len(train_df))

print("\nValidation rows:")
print(len(valid_df))

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
# REMOVE TARGET LEAKAGE
# ============================================================

LEAKAGE_COLUMNS = []

for column in X_train.columns:

    if "posted_rate" in column.lower():
        LEAKAGE_COLUMNS.append(column)

    if column.lower() == "rate_per_mile":
        LEAKAGE_COLUMNS.append(column)


if LEAKAGE_COLUMNS:

    print(
        "\nRemoving leakage columns:"
    )

    for column in LEAKAGE_COLUMNS:
        print(" -", column)

    X_train = X_train.drop(
        columns=LEAKAGE_COLUMNS
    )

    X_valid = X_valid.drop(
        columns=LEAKAGE_COLUMNS
    )


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    "Numerical features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)

print(
    "Total features:",
    len(X_train.columns)
)


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
# XGBOOST
# ============================================================

model = Pipeline(
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
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING REFINED XGBOOST")
print("=" * 70)

model.fit(
    X_train,
    np.log1p(y_train)
)


# ============================================================
# PREDICT
# ============================================================

log_predictions = model.predict(
    X_valid
)

predictions = np.expm1(
    log_predictions
)

predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# METRICS
# ============================================================

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


print("\n" + "=" * 70)
print("REFINED XGBOOST RESULTS")
print("=" * 70)

print(
    f"MAE  : ${mae:,.2f}"
)

print(
    f"RMSE : ${rmse:,.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# BASELINE
# ============================================================

baseline_mae = 135.948341
baseline_rmse = 642.484394
baseline_r2 = 0.822750


print("\n" + "=" * 70)
print("BASELINE VS REFINED")
print("=" * 70)

comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Baseline_XGBoost": [
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ],
    "Refined_XGBoost": [
        mae,
        rmse,
        r2
    ]
})

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# IMPROVEMENT
# ============================================================

print("\n" + "=" * 70)
print("IMPROVEMENT")
print("=" * 70)

mae_improvement = (
    baseline_mae - mae
)

rmse_improvement = (
    baseline_rmse - rmse
)

r2_improvement = (
    r2 - baseline_r2
)

print(
    f"MAE improvement : "
    f"${mae_improvement:,.2f}"
)

print(
    f"RMSE improvement: "
    f"${rmse_improvement:,.2f}"
)

print(
    f"R² improvement  : "
    f"{r2_improvement:.4f}"
)


# ============================================================
# PREDICTION DATA
# ============================================================

prediction_output = valid_df[
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

prediction_output[
    "predicted_rate"
] = predictions

prediction_output[
    "absolute_error"
] = (
    prediction_output["predicted_rate"]
    -
    prediction_output[TARGET]
).abs()

prediction_output[
    "percentage_error"
] = (
    prediction_output["absolute_error"]
    /
    prediction_output[TARGET].replace(
        0,
        np.nan
    )
    * 100
)


# ============================================================
# TOP ERRORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 REFINED XGBOOST ERRORS")
print("=" * 70)

print(
    prediction_output
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR /
    "refined_xgboost_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR /
    "refined_validation_predictions.csv"
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

prediction_output.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 8B COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:",
    OUTPUT_FILE
)

print(
    "Validation predictions saved to:",
    PREDICTIONS_FILE
)