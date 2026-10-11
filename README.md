# ATM Cash Requirement Prediction using Machine Learning

Predicts how much cash an ATM location will need on a given day, so banks can load enough cash to avoid ATMs running dry without tying up excess money.

The project trains and compares four machine learning models on daily ATM withdrawal data, and serves the best one through an interactive Streamlit web app.

## Features

- **Predict a day:** pick an ATM location and a date to get the predicted cash requirement, the recommended amount to load (with an adjustable safety buffer), and a comparison with the actual value when it is available.
- **Forecast ahead:** predicts the next 1 to 14 days with an error band, and lets you download the forecast as CSV.
- **Insights:** weekday patterns, daily trends and top locations.
- **Model performance:** comparison of all models against a simple baseline.

## Dataset

**Wisabi Bank Dataset** from Kaggle (author: obinnaiheanachor), containing ATM transactions from a Nigerian bank.

- 5 states (Enugu, FCT, Kano, Lagos, Rivers), 50 ATM locations
- Year 2022, about 2.14 million transactions in total
- Only withdrawals were used (about 1.19 million), summed per location per day, giving 18,250 daily records
- Amounts are in Nigerian Naira (₦)

> Why a Nigerian dataset? ATM-level transaction data from Indian banks is confidential and not publicly available. The pipeline in this project is not specific to any country and can be retrained on any bank's daily ATM withdrawal data.

## Approach

1. **Data preparation:** combine the five state transaction files, keep withdrawals only, aggregate to daily totals per location, and fill days with no transactions.
2. **Feature engineering:**
   - Calendar: day of week, day of month, weekend flag, salary-week flag (first 7 and last days of the month)
   - History (past days only, to avoid leakage): previous day's withdrawals, same day last week, 7-day rolling average, previous day's transaction count
   - Location: number of ATMs and one-hot encoded location
3. **Time-based split:** the first 80% of dates for training and the last 20% for testing, so the model is never tested on data from before its training period.
4. **Models compared:** Linear Regression, Decision Tree, Random Forest and XGBoost, against a naive baseline ("same as last week").
5. **Deployment:** the best model is saved with joblib and used by the Streamlit app.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost** | 225,895 | 310,874 | 0.473 |
| Random Forest | 226,979 | 311,218 | 0.472 |
| Linear Regression | 237,758 | 323,863 | 0.428 |
| Decision Tree | 236,948 | 332,770 | 0.396 |
| Baseline (same as last week) | 311,663 | 424,138 | 0.019 |

XGBoost and Random Forest perform almost equally. Every model clearly beats the baseline, and XGBoost's MAE is about 27% lower than it. An R² of about 0.47 reflects that withdrawals at a single ATM on a single day are quite noisy.

## Project structure

```
atm-cash-prediction/
├── app.py                      # Streamlit web app
├── requirements.txt
├── notebooks/
│   └── atm_prediction.ipynb    # data prep, features, training, evaluation
├── models/
│   └── atm_model.pkl           # saved XGBoost model and feature list
└── data/
    ├── raw/                    # Kaggle files (not included in the repo)
    └── processed/
        ├── daily_history.csv   # daily withdrawals per location
        └── locations.csv       # location names, cities, ATM counts
```

## How to run

**1. Clone the repository**

```
git clone https://github.com/YOUR-USERNAME/atm-cash-prediction.git
cd atm-cash-prediction
```

**2. Create a virtual environment and install dependencies**

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux, activate with `source venv/bin/activate` instead.

**3. Start the app**

```
streamlit run app.py
```

The app opens at `http://localhost:8501`. The trained model and processed data are included in the repo, so no retraining is needed.

**4. (Optional) Re-run the notebook**

Download the Wisabi Bank Dataset from Kaggle, extract its files into `data/raw/`, then open `notebooks/atm_prediction.ipynb` and run the cells in order. The raw data is not committed to the repo because of its size.

## Limitations and future scope

- Only one year of data, so seasonal and yearly effects cannot be learned.
- Data comes from a Nigerian bank; the model should be retrained on local data before real-world use in another country.
- Multi-day forecasts feed earlier predictions back in as history, so uncertainty grows the further ahead you forecast.
- Future work: holidays and festivals, weather, location-specific events, real-time transaction data, and retraining on Indian bank data when available.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Plotly, Streamlit, joblib

## Team

- T.Niharika
- V.Bhavitha
- A.Praneeth
- Sk.Sameena

Samskruti College of Engineering & Technology- 3/1 - CSE(AIML)
