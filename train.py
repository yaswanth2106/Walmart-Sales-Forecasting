import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

DB_PATH = "walmart.db"
TABLE_NAME = "walmart_sales"



def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df



def make_features(df):
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    iso = df['Date'].dt.isocalendar()
    df['Year'] = iso['year'].astype(int)
    df['Week'] = iso['week'].astype(int)
    df['Month'] = df['Date'].dt.month.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek.astype(int)



    df['Holiday_Flag'] = df['Holiday_Flag'].fillna(0).astype(int)


    numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan


    for lag in range(1, 9):
        df[f"lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)


    df["roll_mean_4"] = (
        df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4, min_periods=1).mean()
    )
    df["roll_std_4"] = (
        df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4, min_periods=1).std()
    )
    df["roll_mean_8"] = (
        df.groupby("Store")["Weekly_Sales"].shift(1).rolling(8, min_periods=1).mean()
    )
    df["roll_std_8"] = (
        df.groupby("Store")["Weekly_Sales"].shift(1).rolling(8, min_periods=1).std()
    )
    df["roll_mean_12"] = (
        df.groupby("Store")["Weekly_Sales"].shift(1).rolling(12, min_periods=1).mean()
    )


    df = df.dropna(subset=["lag_1", "lag_2"]).reset_index(drop=True)
    return df




def time_split(df, test_weeks=12):
    max_date = df["Date"].max()
    cutoff = max_date - pd.Timedelta(weeks=test_weeks)
    train = df[df["Date"] <= cutoff].copy()
    test = df[df["Date"] > cutoff].copy()
    return train, test




def plot_predictions(y_test, preds, store=None):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(preds, label="Predicted")
    title = f"Store {store} – Actual vs Predicted" if store else "Actual vs Predicted"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(y_test, preds):
    residuals = y_test.values - preds
    plt.figure(figsize=(8, 4))
    plt.plot(residuals)
    plt.title("Residuals")
    plt.grid(True)
    plt.show()





def train_per_store(df, feature_cols):

    models = {}
    results = []

    for store_id, store_df in df.groupby("Store"):
        print(f"\n============================")
        print(f"Training LightGBM for Store {store_id}")
        print("============================")

        train_df, test_df = time_split(store_df)

        X_train = train_df[feature_cols]
        y_train = train_df["Weekly_Sales"]
        X_test = test_df[feature_cols]
        y_test = test_df["Weekly_Sales"]

        # LightGBM model
        model = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        

        results.append([store_id, rmse, mae])
        models[store_id] = model

        

    result_df = pd.DataFrame(results, columns=["Store", "RMSE", "MAE"])
    print("\n\n===== FINAL RESULTS =====")
    print(result_df)
    return models, result_df



def forecast_next_week_global(df, model, feature_cols):
    """
    Forecast next week's sales for ALL stores using one global model.
    Saves results to SQLite table: walmart_global_forecast
    """


    latest = df.groupby("Store").tail(1).copy()


    latest["Date"] = latest["Date"] + pd.Timedelta(weeks=1)


    X = latest[feature_cols]


    predictions = model.predict(X)


    result = pd.DataFrame({
        "Store": latest["Store"].values,
        "Date": latest["Date"].values,
        "Predicted_Next_Week": predictions
    })


    conn = sqlite3.connect(DB_PATH)
    result.to_sql("walmart_global_forecast", conn, if_exists="replace", index=False)
    conn.close()

    print("\nSaved next-week predictions to table: walmart_global_forecast")





print("Loading data...")
df = load_data()

print("Engineering features...")
df = make_features(df)

feature_cols = [
    "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Year", "Month", "Week", "DayOfWeek",
    "lag_1","lag_2","lag_3","lag_4",
    "lag_5","lag_6","lag_7","lag_8",
    "roll_mean_4","roll_std_4",
    "roll_mean_8","roll_std_8",
    "roll_mean_12"
]

print("Training one LightGBM model per store...")
models, results = train_per_store(df, feature_cols)

print("Predicting next week for all stores...")



from lightgbm import LGBMRegressor
import joblib
MODEL_PATH = "best_global_lgbm.joblib"  


print("Training GLOBAL LightGBM model...")

train_df, test_df = time_split(df)

X_train = train_df[feature_cols]
y_train = train_df["Weekly_Sales"]
X_test = test_df[feature_cols]
y_test = test_df["Weekly_Sales"]

lgb = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=45,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb.fit(X_train, y_train)

preds = lgb.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print(f"\nGLOBAL MODEL → RMSE: {rmse:.2f}, MAE: {mae:.2f}")




to_save = {
    "model": lgb,
    "feature_cols": feature_cols
}
joblib.dump(to_save, MODEL_PATH)
forecast_next_week_global(df, lgb, feature_cols)

print(f"Saved global model and feature list to {MODEL_PATH}")
