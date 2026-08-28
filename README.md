🚛 Freight Rate Prediction — End-to-End Machine Learning System
Python XGBoost scikit--learn Pandas Status

An end-to-end machine learning pipeline that predicts freight posted_rate from shipment, route, geographic, equipment, market, quote-signal, and temporal information. Built for the Spotter Machine Learning Engineer assessment.

👤 Author
Name	Mahipal Singh Deora
Email	mmahipalsingh717@gmail.com
Mobile	+91 96864 44115
LinkedIn	linkedin.com/in/mahipal-singh-deora
GitHub	github.com/mpsinghmandawariya
🔗 Repository
github.com/mpsinghmandawariya/spotter_assessment

This README is written to live inside the assessment/ folder of the repo (alongside the numbered scripts, data/, and outputs/) — the same location as the folder's own readme.md. Image links and relative paths below assume that placement.

📑 Table of Contents
Project Overview
Dataset
Technology Stack
Original Features
Exploratory Data Analysis
Feature Engineering
Data Quality Handling
Baseline Modeling
Error Analysis
XGBoost Experiments
High-Rate Analysis
Refined Feature Experiment
Feature Importance
Controlled Hyperparameter Tuning
Rolling Time Validation
Robustness Experiment
Final Model
Final Prediction Results
Final Submission Validation
Custom Prediction
Known Limitation
Project Structure
Script Execution Order
How to Run
Evaluation Metrics
Why Time-Based Validation Matters
Final Model Selection Logic
Final Results Summary
Summary
Submission Checklist
Final Files
Final Model at a Glance
Metric	Value
Algorithm	XGBoost Regressor
Target	log1p(posted_rate)
Training rows	48,000
Prediction rows	12,000
Best time-split MAE	$127.14
Best time-split RMSE	$640.12
Best time-split R²	0.8241
Rolling-validation mean MAE	$139.92
Rolling-validation mean RMSE	$636.99
Rolling-validation mean R²	0.8215
Final Deliverables
Priority	File	Purpose
🥇	validation_predictions.csv	Final 12,000-row prediction submission
🥇	outputs/final_model/final_predictions.csv	Final model predictions
🥇	outputs/final_model/final_xgboost_model.pkl	Trained final XGBoost pipeline
🥇	outputs/final_model/submission_audit.csv	Final submission validation/audit
⭐	outputs/final_analysis/model_comparison.csv	Comparison of tested models
⭐	outputs/final_analysis/final_model_config.csv	Final model configuration
⭐	outputs/final_analysis/rolling_validation.csv	Rolling time-validation results
⭐	outputs/feature_importance/aggregated_feature_importance.csv	Aggregated feature importance
⭐	outputs/feature_importance/permutation_importance.csv	Permutation importance
⭐	outputs/error_analysis/validation_error_analysis.csv	Validation error analysis
⭐	21_custom_prediction.py	Run a prediction for a custom shipment
📊	outputs/eda/	EDA charts and summary files
📊	outputs/model_tuning/xgboost_tuning_results.csv	Hyperparameter tuning experiments
1. Project Overview
Freight transportation pricing depends on many interacting factors. The objective of this project is to build a machine learning system that estimates the posted freight rate for a shipment, using inputs such as pickup/delivery location, distance, equipment type, weight, coordinates, date, market index, and quote signal.

The project follows a complete workflow:


2. Dataset
File	Shape	Description
data/train_test.csv	48,000 rows × 14 cols	Labeled training data (includes posted_rate)
data/validation.csv	12,000 rows × 13 cols	Same features, no posted_rate — requires predictions
data/validation-predictions-template.csv	12,000 rows × 2 cols	Submission template (load_id, predicted_rate)
Training columns: load_id, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, distance, equipment, weight, date, market_index, quote_signal, posted_rate

posted_rate is the supervised learning target.

3. Technology Stack
Category	Technology
Programming	Python 3.10
Data Processing	Pandas
Numerical Computing	NumPy
Visualization	Matplotlib
Machine Learning	Scikit-learn
Gradient Boosting	XGBoost
Model Persistence	Joblib
Notebook	Jupyter
Dependencies are listed in requirements.txt.

Note: the repo's requirements.txt currently pins only the packages score.py itself needs (matplotlib, numpy, pandas) for validating and charting a submission. The modeling scripts (03_features.py onward) also import scikit-learn, xgboost, and joblib — add these to requirements.txt (or a requirements-dev.txt) so a reviewer can reproduce training end-to-end with a single pip install.

4. Original Features (14 columns)
Identification: load_id
Locations: pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon
Shipment characteristics: distance, equipment, weight
Time: date
Market information: market_index, quote_signal
Target: posted_rate
5. Exploratory Data Analysis
Scripts: 01_data_audit.py, 02_eda.py — outputs in outputs/eda/

Key finding — distance dominates pricing:

Feature	Correlation with posted_rate
Distance	≈ 0.9085
Weight	≈ 0.0348
Market index	≈ 0.0342
Quote signal	≈ -0.0399
Average posted rate by equipment: Reefer ≈ $2,553.64, Flatbed ≈ $2,445.09, Dry Van ≈ $2,271.55

Monthly trend: highest average monthly rate observed was June 2025 (≈ $2,497.03)

Rate per mile: mean ≈ 2.215, median ≈ 2.145

Distance vs Posted Rate Distance is clearly the dominant driver — the tight lower band is normal-rate loads; the scattered upper cloud is the rare high-rate loads discussed in Section 11.

Average Posted Rate by Equipment

Monthly Average Posted Rate

Additional EDA covered pickup/delivery location frequency, common routes, daily and weekday trends, equipment effects, market index, and quote signal.

6. Feature Engineering
Script: 03_features.py

Group	Features
Date	year, month, day, day_of_week, week_of_year, day_of_year, is_weekend
Cyclic time	month_sin, month_cos, weekday_sin, weekday_cos
Route	Combined pickup → delivery (e.g. Phoenix → Los Angeles)
Distance	distance, distance_log, distance_squared
Weight	weight, weight_log, weight_squared, weight_per_mile
Interaction	distance_weight_interaction
Market	market_index, market_index_squared, market_index_log
Quote	quote_signal, quote_signal_squared, quote_signal_abs
Geographic	lat_difference, lon_difference, geo_distance
7. Data Quality Handling
Feature engineering introduced missing values in market_index, weight, weight_log, weight_squared, weight_per_mile, distance_weight_interaction. No infinite values were found after feature engineering.

Preprocessing pipeline:

Numerical → median imputation
Categorical → most-frequent imputation + one-hot encoding (handle_unknown="ignore" so unseen categories don't break prediction)
8. Baseline Modeling
Script: 04_baseline_models.py

Model	Split	MAE	RMSE	R²
Ridge Regression	Random	$314.37	$654.80	0.7995
Ridge Regression	Time	$322.62	$732.66	0.7695
Random Forest	Random	$120.53	$562.39	0.8521
Random Forest	Time	$157.98	$671.65	0.8063
Time-based results were treated as the more realistic indicator of future performance, since the model needs to predict future loads from historical information.

9. Error Analysis
Script: 05_error_analysis.py — baseline Random Forest analyzed by equipment, distance, month, and largest absolute errors.

Distance Band	Approx. MAE
0–100 mi	$23.79
100–250 mi	$37.77
250–500 mi	$71.67
500–750 mi	$92.28
750–1000 mi	$111.15
1000–1500 mi	$204.29
1500–2500 mi	$264.55
2500+ mi	$269.33

Error grows substantially with distance — long-haul shipments are harder to estimate.

Absolute Prediction Error vs Distance The handful of large outliers are the same rare high-rate loads flagged in the EDA above.

10. XGBoost Experiments
Script: 06_boosting_models.py

Target	MAE	RMSE	R²
Raw	$141.96	$650.33	0.8184
Log-transformed (log1p)	$135.95	$642.48	0.8227
The log-target approach (y_log = np.log1p(y), predictions inverted with np.expm1) performed better and became the foundation of the final model.

11. High-Rate Analysis
Scripts: 07_high_rate_analysis.py, 10_two_stage_model.py, 11_tune_high_rate_strategy.py

The dataset contains very few extreme-rate observations. A two-stage approach (high-rate classifier + XGBoost regressor) improved error on high-rate loads in some cases but degraded overall validation performance, so it was not selected for production. It was retained in the experiment history for transparency.

12. Refined Feature Experiment
Scripts: 08_refined_features.py, 09_test_refined_xgboost.py

Model	MAE	RMSE	R²
Baseline log-target XGBoost	$135.95	$642.48	0.8228
Refined XGBoost (extra interactions)	$138.60	$643.74	0.8221
The refined feature set did not improve generalization and was not adopted — a useful reminder that more features don't automatically help.

13. Feature Importance
Script: 12_feature_importance.py

Top XGBoost features: distance_log, distance, distance_squared, lon_difference, geo_distance

Aggregated importance: distance ≈ 74.9%, route ≈ 14.6%, delivery ≈ 3.7%, pickup ≈ 3.7%


Top permutation-importance signals: distance, lon_difference, distance_log, geo_distance, quote_signal, distance_weight_interaction, weight_per_mile, equipment

Both views reinforce that distance and route characteristics dominate freight-rate prediction.

14. Controlled Hyperparameter Tuning
Script: 13_xgboost_tuning.py

Search space: number of estimators, learning rate, max depth, min child weight, subsample, colsample_bytree.

Best configuration (champion):

n_estimators      = 700
learning_rate     = 0.05
max_depth         = 6
min_child_weight  = 5
subsample         = 0.85
colsample_bytree  = 0.85
Validation performance: MAE $127.14 | RMSE $640.12 | R² 0.8241

15. Rolling Time Validation
Script: 14_rolling_time_validation.py

Window	MAE	RMSE	R²
July	$166.48	$648.15	0.8145
August	$128.31	$622.88	0.8216
September	$122.75	$622.96	0.8328
October	$142.14	$653.97	0.8170
Mean	$139.92	$636.99	0.8215
Stable R² across windows indicates consistent predictive behavior over time, though absolute error varies month to month.

16. Robustness Experiment
Script: 17_robust_xgboost.py

A raw-target XGBoost (MAE $135.42, RMSE $648.38, R² 0.8195) and a Pseudo-Huber loss experiment (rejected — produced invalid/failed results with extremely large errors) were tested. The tuned log-target XGBoost remained the champion.

17. Final Model
Script: 18_final_train.py — retrained on all 48,000 labeled records.

Algorithm           : XGBoost Regressor
Target transform     : log1p(posted_rate)
n_estimators         = 700
learning_rate        = 0.05
max_depth            = 6
min_child_weight     = 5
subsample            = 0.85
colsample_bytree     = 0.85
reg_alpha            = 0.1
reg_lambda           = 1.0
objective            = reg:squarederror
random_state         = 42
18. Final Prediction Results
12,000 predictions generated on data/validation.csv.

Statistic	Value
Minimum	$196.33
Maximum	$6,992.86
Mean	$2,361.03
Median	$2,043.82
P25	$1,270.44
P75	$3,361.45
P90	$4,338.42
P95	$4,922.42
P99	$5,836.09
Quality checks: 0 NaN, 0 infinite, 0 negative, 0 zero predictions, 0 duplicate load IDs.

19. Final Submission Validation
Script: 19_validate_submission.py

✓ 12,000 predictions
✓ Correct columns
✓ Correct load IDs and row order
✓ No missing, infinite, or negative predictions
✓ Template compatible
Primary submission file: validation_predictions.csv (mirrors outputs/final_model/final_predictions.csv)

20. Custom Prediction
Script: 21_custom_prediction.py — loads outputs/final_model/final_xgboost_model.pkl and scores a new shipment without retraining.

Example input: Phoenix → Los Angeles, Dry Van, 370 mi, 30,000 lbs, 2025-10-15, market index 100, quote signal 0.5 Example output: Predicted Freight Rate: $873.28

This is a model prediction for a hypothetical shipment, not an actual market rate, unless a matching labeled observation exists.

21. Known Limitation
The model tends to underpredict rare, extreme-rate shipments, because the training data contains very few examples in that range. Overall MAE is strong, but extreme-rate MAE is noticeably larger. This was explicitly investigated via error analysis, high-rate analysis, two-stage modeling, high-rate strategy tuning, and robustness experiments — not something discovered late or left unaddressed.

Potential future improvements: more historical extreme-rate examples, real-time market features, carrier/lane-specific pricing signals, granular supply-demand indicators, separate modeling of rare high-rate regimes, quantile regression / prediction intervals, model ensembles, and online monitoring with drift detection.

22. Project Structure
assessment/
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
│
├── 01_data_audit.py
├── 02_eda.py
├── 03_features.py
├── 04_baseline_models.py
├── 05_error_analysis.py
├── 06_boosting_models.py
├── 07_high_rate_analysis.py
├── 08_refined_features.py
├── 09_test_refined_xgboost.py
├── 10_two_stage_model.py
├── 11_tune_high_rate_strategy.py
├── 12_feature_importance.py
├── 13_xgboost_tuning.py
├── 14_rolling_time_validation.py
├── 15_residual_analysis.py
├── 17_robust_xgboost.py
├── 18_final_train.py
├── 19_validate_submission.py
├── 20_final_model_summary.py
├── 21_custom_prediction.py
│
├── validation_predictions.csv        ← generated by 19_validate_submission.py (add before final push)
├── scorer_results/
│   └── candidate_december.png        ← generated by score.py (add before final push)
│
├── outputs/
│   ├── eda/
│   ├── error_analysis/
│   ├── feature_importance/
│   ├── final_analysis/
│   ├── final_model/
│   ├── high_rate_analysis/
│   ├── high_rate_tuning/
│   ├── model_tuning/
│   ├── refined_features/
│   ├── residual_analysis/
│   ├── robust_xgboost/
│   ├── rolling_validation/
│   └── two_stage/
│
├── requirements.txt
├── score.py
├── submit.ipynb
└── README.md   (this file — replaces the starter readme.md)
The complete experimentation history is intentionally retained so the process is reproducible and reviewable.

23. Script Execution Order

Not every experiment is required to regenerate the final predictions — the core pipeline is 18_final_train.py → 19_validate_submission.py.

24. How to Run
1. Create environment

python -m venv venv
Activate (Windows PowerShell):

.\venv\Scripts\Activate.ps1
2. Install dependencies

pip install -r requirements.txt
3. Train the final model

python 18_final_train.py
Produces outputs/final_model/final_xgboost_model.pkl, final_predictions.csv, and prediction_summary.csv.

4. Validate the submission

python 19_validate_submission.py
5. Generate final analysis

python 20_final_model_summary.py
6. Run a custom prediction

python 21_custom_prediction.py
7. Validate and generate the required December chart

The repo's score.py is the official Spotter scorer. It validates validation_predictions.csv (12,000 rows, load_id,predicted_rate) and data/december-chart-inputs.csv (31 fixed-route rows, one per December day), then renders the required chart:

python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
This produces scorer_results/candidate_december.png — the fixed December prediction chart required in the written report — and prints a validation summary for both files.

25. Evaluation Metrics
MAE (Mean Absolute Error) — average(|actual - predicted|); average dollar deviation.
RMSE (Root Mean Squared Error) — sqrt(mean((actual - predicted)²)); penalizes large errors more heavily, useful for spotting the effect of extreme-rate misses.
R² (Coefficient of Determination) — 1 - SSE/SST; proportion of target variance explained.
26. Why Time-Based Validation Matters
Freight rates are time-dependent, and a random split can leak overlapping time periods between train and validation. This project therefore trains on earlier dates and validates on later dates (e.g. train 2025-01-01 → 2025-08-31, validate 2025-09-01 → 2025-10-31), plus rolling validation across multiple future windows, to better estimate how the model behaves on future data.

27. Final Model Selection Logic

The champion was chosen for the strongest combination of low MAE, low RMSE, strong R², good time-based performance, stable rolling validation, reasonable complexity, and reproducible training.

28. Final Results Summary
Model	MAE	RMSE	R²
Ridge — Random Split	$314.37	$654.80	0.7995
Random Forest — Random Split	$120.53	$562.39	0.8521
Ridge — Time Split	$322.62	$732.66	0.7695
Random Forest — Time Split	$157.98	$671.65	0.8063
XGBoost — Raw Target	$141.96	$650.33	0.8184
XGBoost — Log Target	$135.95	$642.48	0.8228
Refined XGBoost	$138.60	$643.74	0.8221
Tuned XGBoost — Champion	$127.14	$640.12	0.8241
Raw XGBoost — Robust Experiment	$135.42	$648.38	0.8195
Random Forest performed best on a random split, but this project prioritizes time-aware validation for a future-prediction problem, so the tuned log-target XGBoost model was selected as the final champion.


29. Summary
Built an end-to-end freight rate prediction system in Python using Pandas, NumPy, Scikit-learn, and XGBoost. Started from 48,000 labeled shipment records, ran data auditing and exploratory analysis, and found distance to be the dominant driver of rate — leading to engineered distance, route, geographic, weight, market, quote-signal, and temporal features. Compared Ridge Regression, Random Forest, and XGBoost, evaluated raw vs. log-transformed targets, and — after error analysis, feature-importance analysis, controlled hyperparameter tuning, and rolling time validation — selected the tuned log-target XGBoost model as champion (≈$127 MAE, ≈$640 RMSE, 0.824 R² on time-based validation). The final model was retrained on all 48,000 labeled records and used to generate predictions for the 12,000 unseen loads, with full submission validation and a custom-shipment prediction interface included.

30. Submission Checklist
Verified against the current repo contents:

 outputs/final_model/final_predictions.csv exists (12,000 rows)
 load_id / predicted_rate columns present in the required format
 outputs/final_model/submission_audit.csv confirms no duplicate, missing, infinite, or negative predictions
 Final model saved (outputs/final_model/final_xgboost_model.pkl)
 README included
 Experiment history preserved (all 20 numbered scripts + outputs/)
 Custom prediction script included (21_custom_prediction.py)
 Copy outputs/final_model/final_predictions.csv → validation_predictions.csv at the repo root (not yet present at root in this upload)
 Fill data/december-chart-inputs.csv and run score.py to generate scorer_results/candidate_december.png for the written report
 Expand requirements.txt to include scikit-learn, xgboost, and joblib alongside matplotlib, numpy, pandas, so the full pipeline installs from one file
 PDF/DOCX report (validation approach + December chart) and Loom walkthrough, per the assessment instructions
31. Final Files
validation_predictions.csv

outputs/final_model/
├── final_xgboost_model.pkl
├── final_predictions.csv
├── prediction_summary.csv
└── submission_audit.csv

outputs/final_analysis/
├── final_model_config.csv
├── model_comparison.csv
├── prediction_summary.csv
└── rolling_validation.csv

21_custom_prediction.py
19_validate_submission.py
18_final_train.py
This project delivers a complete freight rate prediction machine learning system — not a single trained model — covering the full lifecycle from data through analysis, feature engineering, baselines, error analysis, model experiments, feature importance, tuning, time validation, final training, prediction, submission validation, and custom inference.

📬 Contact
Mahipal Singh Deora 📧 mmahipalsingh717@gmail.com · 📱 +91 96864 44115 🔗 LinkedIn · 💻 GitHub
