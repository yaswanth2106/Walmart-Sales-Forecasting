# Walmart Sales Forecasting Dashboard

# Project Overview
This repository contains a **Streamlit** dashboard that visualises historical weekly sales for Walmart stores, evaluates a LightGBM forecasting model, and predicts next‑week sales for every store.  The same codebase also includes a **training pipeline** (`train.py`) that engineers features, trains a per‑store LightGBM model as well as a global model, and stores predictions in a SQLite database (`walmart.db`).

---

# Setup & Installation
1. **Clone the repository** (or download the source folder).
2. **Create a virtual environment** (recommended) and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r req.txt
   ```
3. Ensure the SQLite database `walmart.db` is present in the project root (it ships with the repo).

---

# Running the Dashboard
```bash
streamlit run app.py
```
The dashboard will launch in your default browser.  It provides:
- Interactive filters for store and date range.
- Historical weekly sales line chart with optional rolling averages.
- Actual vs. predicted chart for the test set.
- Residuals plot.
- Next‑week forecast metric and a table of forecasts for all stores.
- Feature importance bar chart (top 15 features).
- Monthly sales trend line chart.
- Sales heatmap (store × month).

---

# Training the Model
To re‑train the models and generate fresh forecasts, run:
```bash
python train.py
```
The script will:
1. Load raw data from `walmart.db`.
2. Engineer time‑based and lag features.
3. Train a LightGBM model per store and a **global** model.
4. Save the global model and feature list to `best_global_lgbm.joblib`.
5. Store next‑week predictions in the `walmart_global_forecast` table.

---

# Repository Structure
```
├─ app.py                # Streamlit dashboard
├─ train.py              # Training & forecasting pipeline
├─ req.txt               # Python dependencies (generated)
├─ readme.md             # This documentation
├─ walmart.db            # SQLite database with raw sales data
├─ best_global_lgbm.joblib  # Saved global LightGBM model
└─ train_csv.csv        # Optional CSV export of training data
```

