# Nepal River High-Flow Prediction Using Machine Learning

## Project Overview

This project develops a machine learning pipeline for **predicting
high-flow conditions in rivers of Nepal using hydrological and
meteorological data**.

The dataset contains river discharge, precipitation, rainfall, soil
moisture, temperature, humidity, wind, geographical information, and DHM
station details. The workflow covers exploratory data analysis,
station-specific threshold creation, temporal feature engineering,
class-imbalance handling, machine learning model development, baseline
comparison, evaluation, model saving, and a Streamlit application.

The main objective is to predict whether the **next available river
discharge observation exceeds the station-specific 95th-percentile
high-flow threshold**.

> **Important:** This is an educational/research project and is not
> intended to replace official flood forecasting or warning systems.

## Objectives

-   Explore the structure and characteristics of the hydrological
    dataset.
-   Analyze distributions, missing values, correlations, skewness, and
    outliers.
-   Investigate relationships between rainfall, soil moisture, weather,
    and river discharge.
-   Create station-specific high-flow thresholds using the 95th
    percentile.
-   Engineer temporal, rainfall, soil-moisture, and seasonal features.
-   Prevent temporal leakage through chronological train/test splitting.
-   Compare Logistic Regression, Random Forest, HistGradientBoosting,
    and XGBoost.
-   Handle class imbalance caused by relatively rare high-flow events.
-   Evaluate models using PR-AUC, precision, recall, and F1-score.
-   Compare machine learning models with a persistence baseline.
-   Save the selected model for reuse.
-   Develop an interactive Streamlit prediction application.

## Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost
-   Joblib
-   Streamlit
-   Jupyter Notebook

## Project Structure

``` text
Flood/
├── App/
│   ├── app.py
│   ├── flood_model.joblib
│   └── model_comparison.csv
├── Data/
│   └── dataset.csv
├── notebooks/
│   └── flood_prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The dataset contains **13,390 observations and 19 original variables**.

Important variables include:

  Variable                       Description
  ------------------------------ ------------------------
  `date`                         Observation date
  `dhm_station`                  DHM station identifier
  `elevation_m`                  Station elevation
  `precipitation_mm`             Precipitation
  `soil_moisture_0_100cm_m3m3`   Soil moisture
  `rain_mm`                      Rainfall
  `temperature_mean_c`           Mean temperature
  `relative_humidity_mean_pct`   Mean relative humidity
  `river_discharge_m3s`          River discharge

## Exploratory Data Analysis

The analysis included dataset structure, descriptive statistics, missing
values, duplicates, distributions, skewness, correlations, outlier
investigation, station-level analysis, and temporal patterns.

The dataset initially contained no missing values.

Important skewness values were:

``` text
river_discharge_m3s     ≈ 12.40
rain_mm                 ≈ 7.91
precipitation_mm        ≈ 7.87
```

Extreme hydrological values were not automatically removed because
unusually high rainfall and river discharge can represent genuine
events.

`rain_mm` and `precipitation_mm` were highly correlated:

``` text
Correlation = 0.998538
```

Therefore, `rain_mm` was removed from the initial modeling features to
reduce redundancy.

## Target Variable

For each DHM station, the **95th percentile of river discharge** is
calculated using training-period observations only.

``` python
threshold = training_discharge.quantile(0.95)
```

The target is:

``` text
1 → High-flow condition
0 → Normal-flow condition
```

A station-specific threshold is used because different river stations
naturally have different discharge scales.

## Feature Engineering

The project creates the following important features:

### Discharge Ratio

``` text
discharge_ratio = current discharge / station threshold
```

This makes discharge comparable between stations.

### Previous Discharge

``` text
discharge_yesterday
```

Captures the previous discharge ratio.

### Discharge Change

``` text
discharge_change = current ratio - previous ratio
```

Captures whether river conditions are rising or falling.

### Log Rainfall

``` python
rain_log = np.log1p(precipitation_mm)
```

Reduces the influence of extreme precipitation values.

### Rainfall Accumulation

``` text
rain_3d
rain_7d
```

Represent rainfall accumulated over the previous three and seven
observations.

### Soil Moisture

``` text
soil_3d
```

Represents the moving average of soil moisture over three observations.

### Soil-Rainfall Interaction

``` text
soil_x_rain = soil moisture × rain_3d
```

Represents combined wet-soil and rainfall conditions.

### Seasonal Feature

``` text
month
```

Extracted from the date to capture seasonal effects.

> `rolling(3)` and `rolling(7)` represent observations rather than
> guaranteed calendar days. They correspond to days only when the
> station data is regular and daily.

## Final Feature Set

``` text
discharge_ratio
discharge_yesterday
discharge_change
rain_log
rain_3d
rain_7d
soil_moisture_0_100cm_m3m3
soil_3d
soil_x_rain
temperature_mean_c
relative_humidity_mean_pct
elevation_m
month
```

## Temporal Data Preparation

The observations should be sorted by station and date before shift and
rolling operations:

``` python
df = df.sort_values(["dhm_station", "date"])
```

The project uses a chronological split:

``` text
Past observations   → Training
Future observations → Testing
```

This reduces the risk of temporal data leakage.

## Machine Learning Models

### Logistic Regression

Uses feature scaling and balanced class weights.

``` python
make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
)
```

### Random Forest

``` text
n_estimators = 200
class_weight = balanced
random_state = 42
```

### HistGradientBoosting

``` text
learning_rate = 0.05
max_iter = 300
class_weight = balanced
random_state = 42
```

### XGBoost

``` text
n_estimators = 300
learning_rate = 0.05
max_depth = 4
scale_pos_weight = negative / positive
eval_metric = logloss
random_state = 42
```

## Handling Class Imbalance

High-flow observations are relatively rare.

The project uses:

``` text
class_weight = "balanced"
```

for selected scikit-learn models and:

``` text
scale_pos_weight
```

for XGBoost.

This gives greater importance to the minority high-flow class.

## Persistence Baseline

A simple baseline is also evaluated:

``` python
baseline_pred = (test["discharge_ratio"] > 1).astype(int)
```

It is compared with the machine learning models using PR-AUC, precision,
recall, and F1-score.

## Evaluation Metrics

Because high-flow events are relatively rare, accuracy is not the
primary metric.

The project uses:

-   **PR-AUC** --- important for imbalanced classification.
-   **Precision** --- proportion of predicted high-flow events that are
    correct.
-   **Recall** --- proportion of actual high-flow events detected.
-   **F1-score** --- balance between precision and recall.

## Model Comparison

The models are compared using the same chronological test set.

  ----------------------------------------------------------------------------------
  Model                   PR-AUC         Precision         Recall          F1
  ---------------------- -------------- -------------- -------------- --------------
       
  Random Forest             0.522           0.467           0.625           0.534
          
  HistGradientBoosting      0.507           0.362           0.75            0.488

  XGBoost                   0.518.          0.377           0.821           0.517
  ----------------------------------------------------------------------------------



## Model Saving

The best machine learning model is selected using PR-AUC.

The saved model bundle contains:

``` text
model
model_name
features
station_info
results
```

The main files are:

``` text
flood_model.joblib
model_comparison.csv
```

## Streamlit Web Application

A Streamlit application provides an interactive interface for the
trained high-flow prediction model.

### Application Workflow

``` text
Environmental / Hydrological Data
              ↓
       Feature Preparation
              ↓
 Station-Specific Threshold
              ↓
 Temporal / Rainfall / Soil Features
              ↓
     Trained ML Model
              ↓
    High-Flow Probability
              ↓
     High-Flow Prediction
```

### Run the Application

Activate the virtual environment:

``` bash
source .venv/bin/activate
```

Run Streamlit:

``` bash
python -m streamlit run app.py
```

The application normally opens at:

``` text
http://localhost:8501
```

> **Important:** Do not name the application file `streamlit.py`,
> because it can conflict with the Streamlit package and cause a
> circular-import error.

## Installation

Clone the repository:

``` bash
git clone <your-repository-url>
cd Flood
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it on macOS/Linux:

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the application:

``` bash
python -m streamlit run App/app.py
```

## Usage Workflow

``` text
Data Loading
    ↓
Exploratory Data Analysis
    ↓
Date Conversion and Sorting
    ↓
Station-Specific 95th Percentile
    ↓
Target Creation
    ↓
Feature Engineering
    ↓
Chronological Train/Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Best Model Selection
    ↓
Model Saving
    ↓
Streamlit Deployment
```

## Key Findings

-   River discharge is highly right-skewed because extreme flow events
    are relatively rare.
-   Rainfall and precipitation are almost perfectly correlated
    (`0.998538`).
-   Station-specific thresholds are more appropriate than one absolute
    threshold for all stations.
-   Previous discharge provides useful temporal information.
-   Accumulated rainfall can capture recent hydrological conditions.
-   Soil moisture provides additional information about catchment
    wetness.
-   High-flow prediction is an imbalanced classification problem.
-   PR-AUC is therefore more informative than accuracy alone.
-   Chronological splitting is important for reducing temporal leakage.
-   Tree-based models can capture nonlinear relationships between
    environmental variables.
-   The trained model can be saved and used by the Streamlit
    application.

## Limitations

-   The system uses historical observations rather than a complete
    real-time forecasting pipeline.
-   The 95th percentile is a statistical high-flow definition, not an
    official flood-warning threshold.
-   `rolling(3)` and `rolling(7)` represent observations and depend on
    regular temporal sampling.
-   `shift(-1)` represents the next available observation and is only
    strictly "tomorrow" when observations are daily and continuous.
-   The initial model excludes some geographical and categorical
    variables that may contain useful information.
-   River-network and upstream-downstream relationships are not
    explicitly modeled.
-   Performance may vary between stations and time periods.
-   The model has not been validated as an operational flood-warning
    system.

## Future Improvements

-   Integrate real-time DHM river observations.
-   Integrate real-time and forecasted rainfall.
-   Add upstream/downstream station relationships.
-   Add river-network and geographical features.
-   Experiment with LSTM, GRU, and Transformer models.
-   Add missing-date and temporal-gap validation.
-   Perform hyperparameter optimization.
-   Calibrate prediction probabilities.
-   Develop multiple flood-risk levels instead of binary classification.
-   Add historical discharge and rainfall visualizations.
-   Develop automated warning notifications.
-   Test on additional years and river stations.
-   Perform external validation on unseen stations and periods.

## Results and Reproducibility

The repository should contain:

-   Trained model: `flood_model.joblib`
-   Model comparison: `model_comparison.csv`
-   Streamlit application: `app.py`
-   Analysis notebooks
-   `requirements.txt`
-   `README.md`

Large raw datasets can be excluded from GitHub using `.gitignore`.

Example:

``` text
*.csv
*.joblib
.venv/
__pycache__/
.ipynb_checkpoints/
```

## Author

**Bhijan Bastola**

BSc. CSIT\
Data Science & Machine Learning

## License

This project is intended for educational and research purposes.

## Disclaimer

This project is **not an official flood-warning or emergency-management
system**.

The predictions are intended for educational and research purposes and
should not be used as the sole basis for evacuation decisions, disaster
response, or other safety-critical decisions.
