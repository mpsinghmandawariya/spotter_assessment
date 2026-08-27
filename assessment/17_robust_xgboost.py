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
    "outputs/robust_xgboost"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"


# ============================================================
# BEST HYPERPARAMETERS FROM STEP 12
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 15 - ROBUST RAW-TARGET XGBOOST")
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
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CYCLIC DATE FEATURES
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

    safe_distance = (
        data["distance"]
        .replace(0, np.nan)
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

X_valid = valid_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]

y_valid = valid_df[TARGET]


# ============================================================
# FEATURE TYPES
# ============================================================

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
# MODELS
# ============================================================

models = {

    "Raw XGBoost - Squared Error": {
        "objective": "reg:squarederror"
    },

    "Raw XGBoost - Pseudo Huber": {
        "objective": "reg:pseudohubererror"
    }
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

prediction_store = {}


for model_name, objective_config in models.items():

    print("\n" + "=" * 70)

    print(
        "TRAINING:",
        model_name
    )

    print("=" * 70)

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
                    **objective_config,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    print(
        "\nTraining..."
    )

    # IMPORTANT:
    # Raw target. No log1p transformation.

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_valid
    )

    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXTREME-RATE METRICS
    # --------------------------------------------------------

    extreme_results = {}

    for threshold in [
        5000,
        7500,
        10000,
        12500,
        15000
    ]:

        mask = (
            y_valid >= threshold
        )

        if mask.sum() > 0:

            extreme_mae = mean_absolute_error(
                y_valid[mask],
                predictions[mask]
            )

            extreme_results[
                f"MAE_{threshold}+"
            ] = extreme_mae

        else:

            extreme_results[
                f"MAE_{threshold}+"
            ] = np.nan

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result = {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        **extreme_results
    }

    results.append(
        result
    )

    prediction_store[
        model_name
    ] = predictions

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\nRESULTS")

    print(
        f"MAE  : ${mae:,.2f}"
    )

    print(
        f"RMSE : ${rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        "\nExtreme-rate performance:"
    )

    for threshold in [
        5000,
        7500,
        10000,
        12500,
        15000
    ]:

        key = (
            f"MAE_{threshold}+"
        )

        print(
            f"${threshold:,}+ MAE: "
            f"${extreme_results[key]:,.2f}"
        )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("ROBUST XGBOOST COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# CHAMPION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT CHAMPION")
print("=" * 70)

champion = {
    "model": "Current Log XGBoost",
    "MAE": 127.142833,
    "RMSE": 640.118446,
    "R2": 0.824053
}

comparison_rows = [
    champion
]

for result in results:

    comparison_rows.append({
        "model": result["model"],
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "R2": result["R2"]
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL BY MAE
# ============================================================

best_model = (
    results_df
    .sort_values("MAE")
    .iloc[0]
)

print("\n" + "=" * 70)
print("BEST ROBUST MODEL")
print("=" * 70)

print(
    best_model.to_string()
)


# ============================================================
# DETAILED EXTREME PREDICTIONS
# ============================================================

prediction_df = valid_df[
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


for model_name, predictions in (
    prediction_store.items()
):

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    prediction_df[
        f"{safe_name}_prediction"
    ] = predictions


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "robust_xgboost_results.csv",
    index=False
)

comparison_df.to_csv(
    OUTPUT_DIR /
    "champion_comparison.csv",
    index=False
)

prediction_df.to_csv(
    OUTPUT_DIR /
    "robust_predictions.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 15 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR.resolve()
)