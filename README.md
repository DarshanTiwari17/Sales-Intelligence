# Walmart Sales Intelligence & Forecasting

Simple Zoho-style business dashboard for the **Walmart Recruiting Store Sales Forecasting** dataset (Kaggle).

Story: **Historical Data → Analysis → Model → Prediction**

## Quick start

```bash
pip install -r requirements.txt
python backend/train_model.py   # trains LR, Random Forest, XGBoost (chronological 80/20 split), saves to models/
python app.py                   # open http://127.0.0.1:5000
```

## Using the real Kaggle dataset

1. Download from https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
2. Copy `train.csv`, `features.csv`, `stores.csv` into the `data/` folder (replace the sample files).
3. Re-run `python backend/train_model.py` and restart `python app.py`.

If the CSVs are absent, the app auto-generates realistic sample data with the same schema so the demo still works.

## Pages
- **Overview** — KPIs, weekly trend, top stores/departments
- **Sales Analysis** — filters (store/dept/year/holiday) + time, holiday, monthly, department charts
- **Store Performance** — store selector, summary, trend, department bars, rank #n / 45
- **Sales Prediction** — form (store, dept, date, holiday, temperature, fuel, CPI, unemployment) → expected sales + range
- **Model Performance** — LR / RF / XGBoost compared on MAE, RMSE, R² + actual-vs-predicted scatter

## ML notes (for viva)
- Merged on Store + Date (train + features + stores); date features Year/Month/Week; missing markdowns → 0.
- **No leakage:** Weekly_Sales is never an input feature.
- **Chronological split:** older 80% trains, newer 20% tests (time-series, no shuffling).
