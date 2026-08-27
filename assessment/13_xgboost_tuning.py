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

OUTPUT_DIR = Path(
    "outputs/model_tuning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 12 - CONTROLLED XGBOOST TUNING")
print("=" * 70)

df = pd.read_csv(
    TRAIN_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# FEATURE ENGINEERING
# ORIGINAL SUCCESSFUL FEATURE SET
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    data["year"] = data["date"].dt.year

    data["month"] = (
        data["date"].dt.month
    )

    data["day"] = (
        data["date"].dt.day
    )

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

    # --------------------------------------------------------
    # CYCLIC FEATURES
    # --------------------------------------------------------

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
# TIME-BASED SPLIT
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
# PREPARE DATA
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
# HYPERPARAMETER CONFIGURATIONS
# ============================================================

configs = [

    # --------------------------------------------------------
    # Current baseline
    # --------------------------------------------------------

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    # --------------------------------------------------------
    # Shallower trees
    # --------------------------------------------------------

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    # --------------------------------------------------------
    # Deeper trees
    # --------------------------------------------------------

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 9,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 10,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    # --------------------------------------------------------
    # Lower learning rate + more trees
    # --------------------------------------------------------

    {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    # --------------------------------------------------------
    # Slightly faster learning
    # --------------------------------------------------------

    {
        "n_estimators": 500,
        "learning_rate": 0.07,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.07,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    # --------------------------------------------------------
    # Minimum child weight
    # --------------------------------------------------------

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85
    },

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.95,
        "colsample_bytree": 0.85
    },

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.70
    },

    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 1.00
    }
]


# ============================================================
# TRAIN CONFIGURATIONS
# ============================================================

results = []

print("\n" + "=" * 70)
print("STARTING TUNING")
print("=" * 70)

for i, params in enumerate(
    configs,
    start=1
):

    print(
        f"\n[{i}/{len(configs)}]"
    )

    print(
        params
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                XGBRegressor(
                    **params,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(
        X_train,
        np.log1p(y_train)
    )

    predictions = np.expm1(
        model.predict(X_valid)
    )

    predictions = np.maximum(
        predictions,
        0
    )

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

    result = {
        "config": i,
        **params,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    results.append(
        result
    )

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
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("TOP MODELS BY MAE")
print("=" * 70)

print(
    results_df
    .sort_values("MAE")
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TOP MODELS BY RMSE")
print("=" * 70)

print(
    results_df
    .sort_values("RMSE")
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TOP MODELS BY R²")
print("=" * 70)

print(
    results_df
    .sort_values(
        "R2",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# BEST MODEL
# ============================================================

best_mae = (
    results_df
    .sort_values("MAE")
    .iloc[0]
)

best_rmse = (
    results_df
    .sort_values("RMSE")
    .iloc[0]
)

best_r2 = (
    results_df
    .sort_values(
        "R2",
        ascending=False
    )
    .iloc[0]
)


print("\n" + "=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)

print("\nBest MAE:")

print(
    best_mae.to_string()
)

print("\nBest RMSE:")

print(
    best_rmse.to_string()
)

print("\nBest R²:")

print(
    best_r2.to_string()
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "xgboost_tuning_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR /
    "xgboost_tuning_results.csv"
)