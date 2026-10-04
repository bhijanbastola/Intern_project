
Project Overview
A machine learning-based system for predicting whether a river's discharge will exceed its station-specific high-flow threshold on the following day. The project uses historical hydrological and meteorological data from Nepal and combines exploratory data analysis, feature engineering, machine learning classification, model evaluation, and a Streamlit web application.

Objectives
Analyze historical hydrological and meteorological data.
Identify important factors related to high river discharge.
Create station-specific high-flow thresholds using the 95th percentile.
Develop temporal and rainfall-related features.
Compare multiple machine learning classification models.
Handle class imbalance caused by relatively rare high-flow events.
Evaluate models using PR-AUC, precision, recall, and F1-score.
Save the best-performing model for reuse.
Develop a Streamlit web application.

Dataset
The dataset contains 13,390 observations and 19 original variables, including date, location, river, basin, DHM station, geographical information, precipitation, soil moisture, rainfall, temperature, humidity, wind variables, and river discharge.
Key Feature Engineering
discharge_ratio = current discharge / station threshold
discharge_yesterday = previous discharge ratio
discharge_change = current ratio - previous ratio
rain_log = log1p(precipitation)
rain_3d = three-observation accumulated precipitation
rain_7d = seven-observation accumulated precipitation
soil_3d = three-observation moving average of soil moisture
soil_x_rain = soil moisture × recent rainfall
month = month extracted from date
Machine Learning Models
Logistic Regression with StandardScaler and balanced class weights.
Random Forest with 200 estimators and balanced class weights.
HistGradientBoosting with learning rate 0.05 and 300 iterations.
XGBoost with 300 estimators, learning rate 0.05, max depth 4, and scale_pos_weight.

Evaluation
The models are evaluated using PR-AUC, precision, recall, and F1-score. A persistence baseline is also used for comparison. PR-AUC is emphasized because high-flow events are relatively rare.

Train-Test Strategy
A chronological train-test split is used so that past observations are used for training and future observations are used for testing. This reduces the risk of temporal data leakage.

Saved Files
flood_model.joblib – saved trained model and supporting information.
model_comparison.csv – comparison of model performance.
Streamlit Application
Run the application using:
python -m streamlit run app.py

The application should be available at http://localhost:8501. The application file should not be named streamlit.py because that can conflict with the Streamlit package.
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Joblib
Streamlit
Installation
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python -m streamlit run App/app.py

Possible Project Structure
Flood/

├── App/

│   ├── app.py

│   ├── flood_model.joblib

│   └── model_comparison.csv

├── Data/

│   └── dataset.csv

├── Notebooks/

│   └── flood_prediction.ipynb

├── README.md

└── requirements.txt

Future Improvements
Integrate real-time river discharge data.
Integrate real-time weather data and forecasts.
Include additional DHM stations.
Add geographical and river-network relationships.
Experiment with LSTM and Transformer-based time-series models.
Add prediction probabilities and risk levels.
Add historical discharge and rainfall visualizations.
Develop automated flood-warning notifications.
Perform more extensive hyperparameter tuning.
Validate the model on additional years and stations.

Limitations
This project is primarily a machine learning research and academic project and should not be considered an official flood-warning system. Model performance depends on the quality, coverage, and temporal resolution of the historical data. The rolling features also assume that consecutive observations represent consecutive time periods, so missing dates or irregular observations should be checked before interpreting rolling windows as exact daily measurements.

Author
Bhijan Bastola
BSc. CSIT | Data Science & Machine Learning