"""Data utilities: load real Kaggle CSVs if present, else generate realistic synthetic data."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
STORES_CSV = os.path.join(DATA_DIR, "stores.csv")

np.random.seed(42)

def has_real_data():
    return all(os.path.exists(p) for p in [TRAIN_CSV, FEATURES_CSV, STORES_CSV])

def generate_synthetic_data(n_stores=45, n_depts=10, start="2010-02-05", end="2012-10-26"):
    """Generate Walmart-like synthetic data matching Kaggle schema. Used only when real CSVs are absent."""
    dates = pd.date_range(start=start, end=end, freq="W-FRI")
    store_types = np.random.choice(["A", "B", "C"], size=n_stores, p=[0.35, 0.35, 0.30])
    sizes = np.random.randint(34000, 220000, size=n_stores)
    stores = pd.DataFrame({"Store": range(1, n_stores + 1), "Type": store_types, "Size": sizes})

    holiday_dates = set(pd.to_datetime([
        "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
        "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
        "2012-02-10", "2012-09-07",
    ]).strftime("%Y-%m-%d"))

    feat_rows, train_rows = [], []
    dept_base = {d: np.random.uniform(8000, 45000) for d in range(1, n_depts + 1)}
    for store in range(1, n_stores + 1):
        size_factor = sizes[store - 1] / 100000
        for dt in dates:
            ds = dt.strftime("%Y-%m-%d")
            is_hol = ds in holiday_dates
            temp = 60 + 30 * np.sin(2 * np.pi * dt.dayofyear / 365) + np.random.normal(0, 5)
            fuel = 3.5 + 0.5 * np.sin(2 * np.pi * dt.dayofyear / 365) + np.random.normal(0, 0.15)
            cpi = 170 + (dt.year - 2010) * 5 + np.random.normal(0, 1.5)
            unemp = 8.0 - (dt.year - 2010) * 0.6 + np.random.normal(0, 0.3)
            mds = [max(0, np.random.normal(5000, 8000) if np.random.rand() > 0.6 else 0) for _ in range(5)]
            feat_rows.append([store, ds, temp, fuel] + mds + [cpi, unemp, is_hol])
            month = dt.month
            seasonal = 1 + 0.25 * (month == 12) + 0.18 * (month == 11) + 0.08 * (month == 2)
            for dept in range(1, n_depts + 1):
                base = dept_base[dept] * (0.5 + size_factor * 0.6)
                hol_boost = 1.6 if (is_hol and dept in [1, 2, 5, 8]) else (1.25 if is_hol else 1.0)
                noise = np.random.normal(1.0, 0.12)
                sales = max(500, base * seasonal * hol_boost * noise)
                train_rows.append([store, dept, ds, sales, is_hol])
    features = pd.DataFrame(feat_rows, columns=["Store", "Date", "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "IsHoliday"])
    train = pd.DataFrame(train_rows, columns=["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"])
    return train, features, stores

def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if has_real_data():
        return "real"
    train, features, stores = generate_synthetic_data()
    train.to_csv(TRAIN_CSV, index=False)
    features.to_csv(FEATURES_CSV, index=False)
    stores.to_csv(STORES_CSV, index=False)
    return "synthetic"

def load_merged():
    ensure_data()
    train = pd.read_csv(TRAIN_CSV)
    features = pd.read_csv(FEATURES_CSV)
    stores = pd.read_csv(STORES_CSV)
    for df in (train, features):
        # normalize holiday column (may be bool or string)
        if "IsHoliday" in df.columns:
            df["IsHoliday"] = df["IsHoliday"].astype(str).str.lower().isin(["true", "1", "t"])
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])
    merged = train.merge(features, on=["Store", "Date"], suffixes=("", "_f"), how="left")
    # resolve duplicate IsHoliday (train's wins)
    if "IsHoliday_f" in merged.columns:
        merged.drop(columns=["IsHoliday_f"], inplace=True)
    merged = merged.merge(stores, on="Store", how="left")
    for c in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    for c in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(merged[c].median())
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month
    merged["Week"] = merged["Date"].dt.isocalendar().week.astype(int)
    merged["Type"] = merged["Type"].fillna("A") if "Type" in merged.columns else "A"
    return merged
