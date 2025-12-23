import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
import joblib

DB_PATH = "walmart.db"
MODEL_PATH = "best_global_lgbm.joblib"

st.set_page_config(page_title="Walmart Sales Forecasting Dashboard", layout="wide")



@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM walmart_sales", conn)
    df_pred = pd.read_sql("SELECT * FROM walmart_global_forecast", conn)
    conn.close()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df, df_pred




@st.cache_resource
@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)  # loads dictionary
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    return model, feature_cols





df, df_forecast = load_data()
model, feature_cols = load_model()

stores = sorted(df["Store"].unique())

st.title("📈 Walmart Weekly Sales Forecasting Dashboard")
st.write("Interactive visualization + forecasting using LightGBM global model.")



st.sidebar.header("Filters")

selected_store = st.sidebar.selectbox("Select Store", stores)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df["Date"].min(), df["Date"].max())
)

show_rolling = st.sidebar.checkbox("Show Rolling Averages", value=True)
show_forecast = st.sidebar.checkbox("Show Next-Week Forecast", value=True)



mask = (df["Store"] == selected_store) & \
       (df["Date"] >= pd.to_datetime(date_range[0])) & \
       (df["Date"] <= pd.to_datetime(date_range[1]))
store_df = df[mask].sort_values("Date")




st.subheader(f" Historical Weekly Sales — Store {selected_store}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(store_df["Date"], store_df["Weekly_Sales"], label="Weekly Sales")

if show_rolling:
    ax.plot(store_df["Date"], store_df["Weekly_Sales"].rolling(4).mean(), 
            label="4-week Rolling Mean")
    ax.plot(store_df["Date"], store_df["Weekly_Sales"].rolling(8).mean(), 
            label="8-week Rolling Mean")

ax.set_title(f"Store {selected_store} — Weekly Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)




st.subheader(" Actual vs Predicted — Test Set")


conn = sqlite3.connect(DB_PATH)
test_pred = pd.read_sql("SELECT * FROM walmart_predictions_test", conn)
conn.close()

test_store = test_pred[test_pred["Store"] == selected_store]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test_store["actual"].values, label="Actual")
ax.plot(test_store["predicted"].values, label="Predicted")
ax.set_title(f"Store {selected_store} — Actual vs Predicted")
ax.legend()
st.pyplot(fig)




st.subheader(" Forecast Error (Residuals)")
test_store = test_store.copy()
test_store["residual"] = test_store["actual"] - test_store["predicted"]

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(test_store["residual"].values)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Residuals")
st.pyplot(fig)




if show_forecast:
    st.subheader(" Next-Week Forecast")

    next_row = df_forecast[df_forecast["Store"] == selected_store].iloc[0]
    st.metric(
        label=f"Next Week Predicted Sales for Store {selected_store}",
        value=f"${next_row['Predicted_Next_Week']:,.2f}"
    )

    st.write("### All Stores — Next Week Forecast")
    st.dataframe(df_forecast)




st.subheader(" Feature Importance (LightGBM)")

importance = model.feature_importances_

fi = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
fi = fi.sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=fi.head(15), x="Importance", y="Feature", ax=ax)
ax.set_title("Top 15 Feature Importances")
st.pyplot(fig)




st.subheader(" Monthly Sales Trend")

df["Month"] = df["Date"].dt.to_period("M")
monthly = df[df["Store"] == selected_store].groupby("Month")["Weekly_Sales"].sum()

st.line_chart(monthly)




st.subheader(" Sales Heatmap (Store × Month)")

df["MonthNum"] = df["Date"].dt.month

pivot = df.pivot_table(
    index="Store",
    columns="MonthNum",
    values="Weekly_Sales",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot, cmap="Blues", annot=False, ax=ax)
ax.set_title("Store vs Month Sales Heatmap")
st.pyplot(fig)


st.success("Dashboard loaded successfully!")
