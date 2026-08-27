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
    "outputs/rolling_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"


# ============================================================
# BEST MODEL CONFIGURATION
# ============================================================

BEST_PARAMS = {
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
print("STEP 13 - ROLLING TIME VALIDATION")
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
# VALIDATION WINDOWS
# ============================================================

windows = [

    {
        "name": "July",
        "train_end": "2025-06-30",
        "valid_start": "2025-07-01",
        "valid_end": "2025-07-31"
    },

    {
        "name": "August",
        "train_end": "2025-07-31",
        "valid_start": "2025-08-01",
        "valid_end": "2025-08-31"
    },

    {
        "name": "September",
        "train_end": "2025-08-31",
        "valid_start": "2025-09-01",
        "valid_end": "2025-09-30"
    },

    {
        "name": "October",
        "train_end": "2025-09-30",
        "valid_start": "2025-10-01",
        "valid_end": "2025-10-31"
    }
]


# ============================================================
# RUN VALIDATION
# ============================================================

all_results = []

all_predictions = []


for window in windows:

    print("\n" + "=" * 70)

    print(
        f"VALIDATION WINDOW: {window['name']}"
    )

    print("=" * 70)

    train_end = pd.Timestamp(
        window["train_end"]
    )

    valid_start = pd.Timestamp(
        window["valid_start"]
    )

    valid_end = pd.Timestamp(
        window["valid_end"]
    )

    train_df = df[
        df["date"] <= train_end
    ].copy()

    valid_df = df[
        (
            df["date"] >= valid_start
        )
        &
        (
            df["date"] <= valid_end
        )
    ].copy()

    print(
        "\nTraining rows:",
        len(train_df)
    )

    print(
        "Validation rows:",
        len(valid_df)
    )

    if len(valid_df) == 0:

        print(
            "No validation rows. Skipping."
        )

        continue

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FEATURE TYPES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                XGBRegressor(
                    **BEST_PARAMS,
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
        "\nTraining model..."
    )

    model.fit(
        X_train,
        np.log1p(y_train)
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = np.expm1(
        model.predict(X_valid)
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

    print(
        f"\nMAE  : ${mae:,.2f}"
    )

    print(
        f"RMSE : ${rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    all_results.append({
        "window": window["name"],
        "train_end": window["train_end"],
        "valid_start": window["valid_start"],
        "valid_end": window["valid_end"],
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    temp = valid_df[
        [
            "load_id",
            "date",
            TARGET
        ]
    ].copy()

    temp["predicted_rate"] = (
        predictions
    )

    temp["absolute_error"] = (
        temp[TARGET]
        -
        temp["predicted_rate"]
    ).abs()

    temp["validation_window"] = (
        window["name"]
    )

    all_predictions.append(
        temp
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n" + "=" * 70)
print("ROLLING VALIDATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# AVERAGE PERFORMANCE
# ============================================================

if len(results_df) > 0:

    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE")
    print("=" * 70)

    print(
        f"Mean MAE  : "
        f"${results_df['MAE'].mean():,.2f}"
    )

    print(
        f"Mean RMSE : "
        f"${results_df['RMSE'].mean():,.2f}"
    )

    print(
        f"Mean R²   : "
        f"{results_df['R2'].mean():.4f}"
    )

    print(
        f"MAE std   : "
        f"${results_df['MAE'].std():,.2f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "rolling_validation_results.csv",
    index=False
)


if len(all_predictions) > 0:

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_df.to_csv(
        OUTPUT_DIR /
        "rolling_validation_predictions.csv",
        index=False
    )


print("\n" + "=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR.resolve()
)