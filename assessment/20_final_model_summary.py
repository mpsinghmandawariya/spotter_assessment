from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("outputs/final_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

results = pd.DataFrame([
    {
        "stage": "Ridge - Random Split",
        "MAE": 314.37,
        "RMSE": 654.80,
        "R2": 0.7995
    },
    {
        "stage": "Random Forest - Random Split",
        "MAE": 120.53,
        "RMSE": 562.39,
        "R2": 0.8521
    },
    {
        "stage": "Ridge - Time Split",
        "MAE": 322.62,
        "RMSE": 732.66,
        "R2": 0.7695
    },
    {
        "stage": "Random Forest - Time Split",
        "MAE": 157.98,
        "RMSE": 671.65,
        "R2": 0.8063
    },
    {
        "stage": "XGBoost - Raw Target",
        "MAE": 141.96,
        "RMSE": 650.33,
        "R2": 0.8184
    },
    {
        "stage": "XGBoost - Log Target",
        "MAE": 135.95,
        "RMSE": 642.48,
        "R2": 0.8228
    },
    {
        "stage": "Refined XGBoost",
        "MAE": 138.60,
        "RMSE": 643.74,
        "R2": 0.8221
    },
    {
        "stage": "Tuned XGBoost",
        "MAE": 127.14,
        "RMSE": 640.12,
        "R2": 0.8241
    },
    {
        "stage": "Raw XGBoost Final",
        "MAE": 135.42,
        "RMSE": 648.38,
        "R2": 0.8195
    }
])


# ============================================================
# SORT BY MAE
# ============================================================

results = results.sort_values(
    "MAE"
).reset_index(drop=True)


print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(index=False)
)


# ============================================================
# BEST MODEL
# ============================================================

best = results.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model : {best['stage']}"
)

print(
    f"MAE   : ${best['MAE']:,.2f}"
)

print(
    f"RMSE  : ${best['RMSE']:,.2f}"
)

print(
    f"R²    : {best['R2']:.4f}"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

config = pd.DataFrame([
    ["Algorithm", "XGBoost Regressor"],
    ["Target transformation", "log1p(posted_rate)"],
    ["n_estimators", 700],
    ["learning_rate", 0.05],
    ["max_depth", 6],
    ["min_child_weight", 5],
    ["subsample", 0.85],
    ["colsample_bytree", 0.85],
    ["reg_alpha", 0.1],
    ["reg_lambda", 1.0],
    ["Training rows", 48000],
    ["Prediction rows", 12000]
], columns=["parameter", "value"])


print("\n" + "=" * 70)
print("FINAL MODEL CONFIGURATION")
print("=" * 70)

print(
    config.to_string(index=False)
)


# ============================================================
# ROLLING VALIDATION
# ============================================================

rolling = pd.DataFrame([
    {
        "window": "July",
        "MAE": 166.48,
        "RMSE": 648.15,
        "R2": 0.8145
    },
    {
        "window": "August",
        "MAE": 128.31,
        "RMSE": 622.88,
        "R2": 0.8216
    },
    {
        "window": "September",
        "MAE": 122.75,
        "RMSE": 622.96,
        "R2": 0.8328
    },
    {
        "window": "October",
        "MAE": 142.14,
        "RMSE": 653.97,
        "R2": 0.8170
    }
])


print("\n" + "=" * 70)
print("ROLLING VALIDATION")
print("=" * 70)

print(
    rolling.to_string(index=False)
)


# ============================================================
# FINAL PREDICTION SUMMARY
# ============================================================

prediction_file = Path(
    "outputs/final_model/final_predictions.csv"
)

predictions = pd.read_csv(
    prediction_file
)

p = predictions["predicted_rate"]

prediction_summary = pd.DataFrame([
    ["Count", len(p)],
    ["Minimum", p.min()],
    ["Maximum", p.max()],
    ["Mean", p.mean()],
    ["Median", p.median()],
    ["P25", p.quantile(0.25)],
    ["P75", p.quantile(0.75)],
    ["P95", p.quantile(0.95)],
    ["P99", p.quantile(0.99)]
], columns=["metric", "value"])


print("\n" + "=" * 70)
print("FINAL PREDICTION SUMMARY")
print("=" * 70)

print(
    prediction_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

config.to_csv(
    OUTPUT_DIR / "final_model_config.csv",
    index=False
)

rolling.to_csv(
    OUTPUT_DIR / "rolling_validation.csv",
    index=False
)

prediction_summary.to_csv(
    OUTPUT_DIR / "prediction_summary.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 18 COMPLETE")
print("=" * 70)

print(
    "\nFinal analysis files saved to:"
)

print(
    OUTPUT_DIR.resolve()
)