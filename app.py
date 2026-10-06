"""Flask backend: dashboard APIs + prediction. Zoho-style simple business app."""
import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from data_loader import load_merged, has_real_data

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = BASE  # app.py lives at project root
MODEL_DIR = os.path.join(ROOT, "models")

app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"),
            static_folder=os.path.join(ROOT, "static"))

DF = load_merged()
METRICS = {}
if os.path.exists(os.path.join(MODEL_DIR, "metrics.json")):
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        METRICS = json.load(f)

LOADED_MODELS = {}
def get_model(name):
    if name in LOADED_MODELS:
        return LOADED_MODELS[name]
    path = os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            LOADED_MODELS[name] = pickle.load(f)
            return LOADED_MODELS[name]
    return None

def encode_row(payload, columns):
    row = {c: 0 for c in columns}
    type_cols = [c for c in columns if c.startswith("Type_")]
    t = str(payload.get("store_type", "A"))
    for c in type_cols:
        row[c] = 1 if c == f"Type_{t}" else 0
    mapping = {"Store": "store", "Dept": "dept", "Temperature": "temperature",
        "Fuel_Price": "fuel_price", "MarkDown1": "md1", "MarkDown2": "md2",
        "MarkDown3": "md3", "MarkDown4": "md4", "MarkDown5": "md5",
        "CPI": "cpi", "Unemployment": "unemployment", "IsHoliday": "is_holiday",
        "Size": "size", "Year": "year", "Month": "month", "Week": "week"}
    for col, key in mapping.items():
        if col in row:
            v = payload.get(key, 0)
            row[col] = int(v) if col == "IsHoliday" else float(v)
    return pd.DataFrame([[row[c] for c in columns]], columns=columns)

# ---------- Pages ----------
@app.route("/")
def overview(): return render_template("overview.html")
@app.route("/analysis")
def analysis(): return render_template("analysis.html")
@app.route("/stores")
def stores(): return render_template("stores.html")
@app.route("/predict")
def predict_page(): return render_template("predict.html")
@app.route("/performance")
def performance(): return render_template("performance.html")

# ---------- APIs ----------
@app.route("/api/meta")
def meta():
    return jsonify({
        "stores": sorted(DF["Store"].unique().tolist()),
        "depts": sorted(DF["Dept"].unique().tolist()),
        "years": sorted(DF["Year"].unique().tolist()),
        "data_source": "real Kaggle CSVs" if has_real_data() else "built-in sample data (add train.csv, features.csv, stores.csv to /data for real Kaggle data)",
        "date_min": DF["Date"].min().strftime("%Y-%m-%d"),
        "date_max": DF["Date"].max().strftime("%Y-%m-%d"),
    })

@app.route("/api/kpis")
def kpis():
    return jsonify({
        "total_sales": float(DF["Weekly_Sales"].sum()),
        "avg_weekly_sales": float(DF.groupby("Date")["Weekly_Sales"].sum().mean()),
        "n_stores": int(DF["Store"].nunique()),
        "n_depts": int(DF["Dept"].nunique()),
    })

@app.route("/api/weekly_trend")
def weekly_trend():
    g = DF.groupby("Date")["Weekly_Sales"].sum().reset_index().sort_values("Date")
    return jsonify({"dates": g["Date"].dt.strftime("%Y-%m-%d").tolist(),
                    "sales": g["Weekly_Sales"].round(2).tolist()})

@app.route("/api/top")
def top():
    s = DF.groupby("Store")["Weekly_Sales"].agg(["sum", "mean"]).reset_index().sort_values("sum", ascending=False)
    d = DF.groupby("Dept")["Weekly_Sales"].agg(["sum", "mean"]).reset_index().sort_values("sum", ascending=False)
    return jsonify({
        "stores": [{"store": int(r.Store), "total": round(float(r["sum"]), 2), "avg": round(float(r["mean"]), 2)} for r in s.itertuples()],
        "depts": [{"dept": int(r.Dept), "total": round(float(r["sum"]), 2), "avg": round(float(r["mean"]), 2)} for r in d.itertuples()],
    })

@app.route("/api/analysis")
def analysis_api():
    store = request.args.get("store", "all")
    dept = request.args.get("dept", "all")
    year = request.args.get("year", "all")
    hol = request.args.get("holiday", "all")
    f = DF
    if store != "all": f = f[f["Store"] == int(store)]
    if dept != "all": f = f[f["Dept"] == int(dept)]
    if year != "all": f = f[f["Year"] == int(year)]
    if hol == "yes": f = f[f["IsHoliday"] == True]
    elif hol == "no": f = f[f["IsHoliday"] == False]
    trend = f.groupby("Date")["Weekly_Sales"].sum().reset_index().sort_values("Date")
    hol_g = DF.groupby(DF["IsHoliday"].map({True: "Holiday", False: "Non-Holiday"}))["Weekly_Sales"].sum()
    monthly = f.groupby("Month")["Weekly_Sales"].sum().reset_index().sort_values("Month")
    bydept = f.groupby("Dept")["Weekly_Sales"].sum().reset_index().sort_values("Weekly_Sales", ascending=True)
    return jsonify({
        "trend": {"dates": trend["Date"].dt.strftime("%Y-%m-%d").tolist(), "sales": trend["Weekly_Sales"].round(2).tolist()},
        "holiday": {"labels": hol_g.index.tolist(), "sales": hol_g.round(2).tolist()},
        "monthly": {"labels": [f"Month {int(m)}" for m in monthly["Month"]], "sales": monthly["Weekly_Sales"].round(2).tolist()},
        "bydept": {"labels": [f"Dept {int(d)}" for d in bydept["Dept"]], "sales": bydept["Weekly_Sales"].round(2).tolist()},
        "count": int(len(f)),
    })

@app.route("/api/store/<int:sid>")
def store_detail(sid):
    f = DF[DF["Store"] == sid]
    if f.empty: return jsonify({"error": "store not found"}), 404
    stype = str(f["Type"].iloc[0]); size = int(f["Size"].iloc[0])
    total = float(f["Weekly_Sales"].sum()); avg = float(f.groupby("Date")["Weekly_Sales"].sum().mean())
    bd = f.groupby("Dept")["Weekly_Sales"].sum().idxmax()
    best_week = f.groupby("Date")["Weekly_Sales"].sum().idxmax()
    trend = f.groupby("Date")["Weekly_Sales"].sum().reset_index().sort_values("Date")
    bydept = f.groupby("Dept")["Weekly_Sales"].sum().reset_index().sort_values("Dept")
    totals = DF.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
    rank = int(totals.index.get_loc(sid)) + 1
    return jsonify({
        "store": sid, "type": stype, "size": size, "total": round(total, 2),
        "avg_weekly": round(avg, 2), "best_dept": int(bd),
        "best_week": pd.to_datetime(best_week).strftime("%Y-%m-%d"),
        "trend": {"dates": trend["Date"].dt.strftime("%Y-%m-%d").tolist(), "sales": trend["Weekly_Sales"].round(2).tolist()},
        "bydept": {"labels": [f"Dept {int(d)}" for d in bydept["Dept"]], "sales": bydept["Weekly_Sales"].round(2).tolist()},
        "rank": rank, "n_stores": int(DF["Store"].nunique()),
    })

@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS if METRICS else {"metrics": {}, "best_model": None})

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    if not METRICS:
        return jsonify({"error": "Models not trained yet. Run: python backend/train_model.py"}), 400
    best = METRICS.get("best_model", "Random Forest")
    model = get_model(body.get("model", best))
    if model is None:
        return jsonify({"error": "Model file not found. Run training first."}), 400
    cols = METRICS["columns"]
    # fill store size/type + date parts
    sid = int(body.get("store", 1))
    srow = DF[DF["Store"] == sid]
    payload = {
        "store": sid, "dept": int(body.get("dept", 1)),
        "temperature": float(body.get("temperature", 65)),
        "fuel_price": float(body.get("fuel_price", 3.6)),
        "md1": float(body.get("md1", 0)), "md2": float(body.get("md2", 0)),
        "md3": float(body.get("md3", 0)), "md4": float(body.get("md4", 0)),
        "md5": float(body.get("md5", 0)),
        "cpi": float(body.get("cpi", 175)), "unemployment": float(body.get("unemployment", 7.5)),
        "is_holiday": 1 if str(body.get("is_holiday", "no")).lower() in ("1", "yes", "true") else 0,
        "store_type": str(srow["Type"].iloc[0]) if len(srow) else "A",
        "size": float(srow["Size"].iloc[0]) if len(srow) else 150000.0,
    }
    try:
        dt = pd.to_datetime(body.get("date", DF["Date"].max()))
    except Exception:
        dt = DF["Date"].max()
    payload["year"] = dt.year; payload["month"] = dt.month
    payload["week"] = int(dt.isocalendar().week)
    X = encode_row(payload, cols)
    pred = float(model.predict(X)[0])
    rmse = METRICS["metrics"].get(best, {}).get("RMSE", pred * 0.1)
    lo, hi = max(0, pred - rmse), pred + rmse
    return jsonify({"prediction": round(pred, 2), "low": round(lo, 2), "high": round(hi, 2),
                    "model": body.get("model", best)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
