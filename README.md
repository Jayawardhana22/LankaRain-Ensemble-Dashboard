# LankaRain Ensemble Lab — ML & Pattern Recognition Assignment

## Project idea
Predict whether rainfall on the next day in Sri Lanka is at least 1 mm using current-day weather, recent rainfall/weather lags, city, and seasonal date features. This is a classification demonstration, not an official forecast.

## Dataset
The project downloads `weather.csv` from Lanka Data Tracker when the local file is absent:
https://github.com/dhananjaya-hbc/lanka-data-tracker
The dataset tracks daily Sri Lankan cities using Open-Meteo data. The collector repository documents its weather data, cities, columns, and historical backfill. Check the dataset file and README at the source before final submission for current schema/availability.
Direct CSV: https://raw.githubusercontent.com/dhananjaya-hbc/lanka-data-tracker/main/data/weather.csv

## Ensemble methods
- Bagging: Random Forest
- Boosting: Gradient Boosting classifier
- Stacking: RF, Gradient Boosting, Extra Trees base estimators + Logistic Regression meta-estimator
- Hybrid: soft-voting Random Forest + Gradient Boosting

## Metrics
Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC. The chronological test set is the latest 20% of target dates; the project avoids random splitting to reduce future-date leakage.

## Windows setup (Command Prompt)
1. Install Python 3.10+ (during install, enable **Add Python to PATH**).
2. Download/extract this project ZIP.
3. Open Command Prompt and change into the extracted folder. Example:
   `cd %USERPROFILE%\Downloads\ensemble_sri_lanka_dashboard`
4. Create a virtual environment:
   `python -m venv .venv`
5. Activate it:
   `.venv\Scripts\activate`
6. Install dependencies:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
7. Download dataset and train models:
   `python train.py`
8. Start dashboard:
   `streamlit run app.py`
9. Browser should open Streamlit. If not, open the localhost URL printed in CMD (usually `http://localhost:8501` or `http://localhost:8501`).

## Notebook
Open `Ensemble_Learning_Assignment.ipynb` in VS Code or Jupyter. Run cells from top to bottom. It includes dataset inspection, class balance, model training, metric comparison, and interpretation prompts.


## Suggested project structure
```
ensemble_sri_lanka_dashboard/
  app.py
  ml_core.py
  train.py
  Ensemble_Learning_Assignment.ipynb
  requirements.txt
  README.md
  VIDEO_SCRIPT.md
  data/        (created after download)
  artifacts/    (created after model training)
```

## AI-use disclosure
AI tools assisted in generating starter code, debugging suggestions, and UI structure. The student should review, run, modify, and understand the code, then report actual dataset details and measured results. Do not claim results that were not produced by the run.

