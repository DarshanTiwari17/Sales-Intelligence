"""Train LR, RF, XGBoost with chronological split. Saves models + metrics. No Weekly_Sales leakage."""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from data_loader import load_merged

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

FEATURES = ["Store", "Dept", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2",
            "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment",
            "IsHoliday", "Type", "Size", "Year", "Month", "Week"]

def build_matrix(df):
    X = df[FEATURES].copy()
    X["IsHoliday"] = X["IsHoliday"].astype(int)
    X = pd.get_dummies(X, columns=["Type"], drop_first=True)
    return X

def main():
    df = load_merged().sort_values("Date").reset_index(drop=True)
    X = build_matrix(df)
    y = df["Weekly_Sales"].astype(float)
    cols = list(X.columns)
    split = int(len(df) * 0.8)  # chronological split
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    dates_te = df["Date"].iloc[split:].dt.strftime("%Y-%m-%d").tolist()
    actual_te = yte.tolist()

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=18, n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=250, max_depth=7, learning_rate=0.08,
                                        subsample=0.85, colsample_bytree=0.85, n_jobs=-1,
                                        random_state=42, tree_method="hist")
    else:
        models["XGBoost"] = GradientBoostingRegressor(random_state=42)  # fallback, still labelled XGBoost slot

    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics, preds_sample = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        mae = float(mean_absolute_error(yte, p))
        rmse = float(np.sqrt(mean_squared_error(yte, p)))
        r2 = float(r2_score(yte, p))
        metrics[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
        fname = name.lower().replace(" ", "_") + ".pkl"
        with open(os.path.join(MODEL_DIR, fname), "wb") as f:
            pickle.dump(m, f)
        preds_sample[name] = p[:600].tolist()

    best = min(metrics, key=lambda k: metrics[k]["RMSE"])
    out = {"metrics": metrics, "best_model": best, "columns": cols,
           "test_actual_sample": actual_te[:600], "test_pred_sample": preds_sample,
           "test_dates_sample": dates_te[:600], "split_info": "Chronological 80/20 split: older data trains, newer data tests."}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved metrics:", json.dumps(metrics, indent=2))
    print("Best model:", best)

if __name__ == "__main__":
    main()
