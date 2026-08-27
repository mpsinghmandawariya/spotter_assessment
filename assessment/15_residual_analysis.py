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
# CONFIG
# ============================================================

DATA_DIR = Path("data")

OUTPUT_DIR = Path(
    "outputs/residual_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"


# ============================================================
# BEST MODEL
# ============================================================

PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85
}


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 14 - RESIDUAL & EXTREME-RATE ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    TRAIN_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# FEATURES
# ============================================================

def create_features(data):

    data = data.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day

    data["day_of_week"] = (
        data["date"].dt.dayofweek
    )

    data["week_of_year"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["day_of_year"] = (
        data["date"].dt.dayofyear
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

    data["route"] = (
        data["pickup"].astype(str)
        + " → "
        + data["delivery"].astype(str)
    )

    data["distance_log"] = np.log1p(
        data["distance"].clip(lower=0)
    )

    data["distance_squared"] = (
        data["distance"] ** 2
    )

    data["weight_log"] = np.log1p(
        data["weight"].clip(lower=0)
    )

    data["weight_squared"] = (
        data["weight"] ** 2
    )

    safe_distance = data[
        "distance"
    ].replace(
        0,
        np.nan
    )

    data["weight_per_mile"] = (
        data["weight"]
        /
        safe_distance
    )

    data["distance_weight_interaction"] = (
        data["distance"]
        *
        data["weight"]
    )

    data["market_index_squared"] = (
        data["market_index"] ** 2
    )

    data["market_index_log"] = np.log1p(
        data["market_index"].clip(lower=0)
    )

    data["quote_signal_squared"] = (
        data["quote_signal"] ** 2
    )

    data["quote_signal_abs"] = (
        data["quote_signal"].abs()
    )

    data["lat_difference"] = (
        data["delivery_lat"]
        -
        data["pickup_lat"]
    ).abs()

    data["lon_difference"] = (
        data["delivery_lon"]
        -
        data["pickup_lon"]
    ).abs()

    data["geo_distance"] = np.sqrt(
        (
            data["delivery_lat"]
            -
            data["pickup_lat"]
        ) ** 2
        +
        (
            data["delivery_lon"]
            -
            data["pickup_lon"]
        ) ** 2
    )

    return data


df = create_features(df)


# ============================================================
# TIME SPLIT
# ============================================================

train_df = df[
    df["date"] < SPLIT_DATE
].copy()

valid_df = df[
    df["date"] >= SPLIT_DATE
].copy()


print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(valid_df)
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

X_valid = valid_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]

y_valid = valid_df[TARGET]


categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)

numeric_features = (
    X_train
    .select_dtypes(
        include=[
            "number",
            "bool"
        ]
    )
    .columns
    .tolist()
)


# ============================================================
# PREPROCESSOR
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
# MODEL
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
                **PARAMS,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print(
    "\nTraining final candidate model..."
)

model.fit(
    X_train,
    np.log1p(y_train)
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = np.expm1(
    model.predict(X_valid)
)

predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# RESIDUAL DATASET
# ============================================================

residuals = valid_df[
    [
        "load_id",
        "date",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        TARGET
    ]
].copy()

residuals["predicted_rate"] = (
    predictions
)

residuals["error"] = (
    residuals["predicted_rate"]
    -
    residuals[TARGET]
)

residuals["absolute_error"] = (
    residuals["error"].abs()
)

residuals["percentage_error"] = (
    residuals["absolute_error"]
    /
    residuals[TARGET]
    *
    100
)

residuals["underpredicted"] = (
    residuals["error"] < 0
)

residuals["month"] = (
    residuals["date"].dt.month
)

residuals["distance_bucket"] = pd.cut(
    residuals["distance"],
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


# ============================================================
# OVERALL
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
print("OVERALL PERFORMANCE")
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
# BIAS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION BIAS")
print("=" * 70)

mean_error = residuals[
    "error"
].mean()

median_error = residuals[
    "error"
].median()

underprediction_rate = (
    residuals["underpredicted"]
    .mean()
    * 100
)

print(
    f"Mean error     : ${mean_error:,.2f}"
)

print(
    f"Median error   : ${median_error:,.2f}"
)

print(
    f"Underprediction: "
    f"{underprediction_rate:.2f}%"
)


# ============================================================
# EXTREME RATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXTREME RATE ANALYSIS")
print("=" * 70)

thresholds = [
    5000,
    7500,
    10000,
    12500,
    15000
]

extreme_results = []

for threshold in thresholds:

    mask = (
        residuals[TARGET]
        >= threshold
    )

    count = mask.sum()

    if count == 0:
        continue

    extreme_mae = mean_absolute_error(
        residuals.loc[mask, TARGET],
        residuals.loc[
            mask,
            "predicted_rate"
        ]
    )

    mean_actual = (
        residuals.loc[
            mask,
            TARGET
        ].mean()
    )

    mean_prediction = (
        residuals.loc[
            mask,
            "predicted_rate"
        ].mean()
    )

    extreme_results.append({
        "threshold": threshold,
        "count": count,
        "MAE": extreme_mae,
        "mean_actual": mean_actual,
        "mean_prediction": mean_prediction
    })

extreme_df = pd.DataFrame(
    extreme_results
)

print(
    extreme_df.to_string(
        index=False
    )
)


# ============================================================
# ERROR BY DISTANCE
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY DISTANCE")
print("=" * 70)

distance_results = (
    residuals
    .groupby(
        "distance_bucket",
        observed=True
    )
    .agg(
        count=(
            TARGET,
            "count"
        ),
        MAE=(
            "absolute_error",
            "mean"
        ),
        RMSE=(
            "absolute_error",
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        )
    )
    .reset_index()
)

print(
    distance_results.to_string(
        index=False
    )
)


# ============================================================
# ERROR BY EQUIPMENT
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY EQUIPMENT")
print("=" * 70)

equipment_results = (
    residuals
    .groupby("equipment")
    .agg(
        count=(
            TARGET,
            "count"
        ),
        MAE=(
            "absolute_error",
            "mean"
        ),
        mean_actual=(
            TARGET,
            "mean"
        ),
        mean_prediction=(
            "predicted_rate",
            "mean"
        )
    )
    .reset_index()
)

print(
    equipment_results.to_string(
        index=False
    )
)


# ============================================================
# ERROR BY MONTH
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY MONTH")
print("=" * 70)

month_results = (
    residuals
    .groupby("month")
    .agg(
        count=(
            TARGET,
            "count"
        ),
        MAE=(
            "absolute_error",
            "mean"
        ),
        mean_actual=(
            TARGET,
            "mean"
        ),
        mean_prediction=(
            "predicted_rate",
            "mean"
        )
    )
    .reset_index()
)

print(
    month_results.to_string(
        index=False
    )
)


# ============================================================
# TOP ROUTES BY ERROR
# ============================================================

print("\n" + "=" * 70)
print("ROUTES WITH LARGEST ERRORS")
print("=" * 70)

route_results = (
    residuals
    .groupby(
        [
            "pickup",
            "delivery"
        ]
    )
    .agg(
        count=(
            TARGET,
            "count"
        ),
        MAE=(
            "absolute_error",
            "mean"
        ),
        mean_actual=(
            TARGET,
            "mean"
        ),
        mean_prediction=(
            "predicted_rate",
            "mean"
        )
    )
    .reset_index()
)

route_results = route_results[
    route_results["count"] >= 5
]

route_results = route_results.sort_values(
    "MAE",
    ascending=False
)

print(
    route_results
    .head(30)
    .to_string(index=False)
)


# ============================================================
# TOP INDIVIDUAL ERRORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 INDIVIDUAL ERRORS")
print("=" * 70)

print(
    residuals
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

residuals.to_csv(
    OUTPUT_DIR /
    "validation_residuals.csv",
    index=False
)

extreme_df.to_csv(
    OUTPUT_DIR /
    "extreme_rate_analysis.csv",
    index=False
)

distance_results.to_csv(
    OUTPUT_DIR /
    "distance_error_analysis.csv",
    index=False
)

equipment_results.to_csv(
    OUTPUT_DIR /
    "equipment_error_analysis.csv",
    index=False
)

month_results.to_csv(
    OUTPUT_DIR /
    "monthly_error_analysis.csv",
    index=False
)

route_results.to_csv(
    OUTPUT_DIR /
    "route_error_analysis.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR.resolve()
)