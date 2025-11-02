# fx_app_full.py
"""
Full-featured FX Forecasting Dashboard (single-file)
- Professional layout + FinTech live data features
- Multi-currency support (file or live)
- Theme switcher (Light / Dark / Corporate)
- Plotly interactive visuals
- Models: ARIMA, SARIMA, Prophet, XGBoost, LSTM, CatBoost (toggleable)
- Deterministic AI Insights (no external LLM calls)
- Download metrics & forecasts CSV
- Optional live rates via exchangerate.host (no API key)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import warnings
warnings.filterwarnings("ignore")

# ML & TS libs (import when models run)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from catboost import CatBoostRegressor

# ---------------------------
# Page & theme
# ---------------------------
st.set_page_config(page_title="FX Forecast Dashboard — Full", layout="wide", initial_sidebar_state="expanded")

THEMES = {
    "Light": {"bg": "#FFFFFF", "fg": "#0f172a"},
    "Dark": {"bg": "#0b1220", "fg": "#E6EEF8"},
    "Corporate Blue": {"bg": "#F3F7FB", "fg": "#073B4C"}
}

# Sidebar — settings
st.sidebar.header("Data & Settings")

theme_choice = st.sidebar.selectbox("Theme", options=["Light", "Dark", "Corporate Blue"])
theme = THEMES[theme_choice]

# Data input modes
data_mode = st.sidebar.radio("Data source", ["Upload Excel/CSV (multi-currency)", "Use sample file in folder", "Fetch live rates (API)"])

uploaded = None
if data_mode == "Upload Excel/CSV (multi-currency)":
    uploaded = st.sidebar.file_uploader("Upload file (.xlsx/.csv)", type=["xlsx","xls","csv"])
elif data_mode == "Use sample file in folder":
    st.sidebar.write("Using `Foreign_Exchange_Rates[1].xlsx` in app folder (make sure it's there).")
else: # live
    st.sidebar.write("Will fetch latest rates from exchangerate.host (free, no API key).")

# Multi-currency UI
multi_currency = st.sidebar.checkbox("Enable multi-currency comparison", value=True)

# Live fetch controls
if data_mode == "Fetch live rates (API)":
    base_currency = st.sidebar.text_input("Base currency (3-letter)", value="USD").upper()
    symbols_input = st.sidebar.text_input("Target currencies (comma separated)", value="EUR,GBP,JPY")
    live_days = st.sidebar.number_input("Days of historical data to fetch (max 365)", min_value=1, max_value=365, value=180)

# Model toggles
st.sidebar.markdown("---")
st.sidebar.write("Model training toggles")
use_arima = st.sidebar.checkbox("ARIMA", True)
use_sarima = st.sidebar.checkbox("SARIMA", True)
use_prophet = st.sidebar.checkbox("Prophet", True)
use_xgb = st.sidebar.checkbox("XGBoost", True)
use_lstm = st.sidebar.checkbox("LSTM", True)
use_cat = st.sidebar.checkbox("CatBoost", True)

# Quick run
st.sidebar.markdown("---")
quick_run = st.sidebar.checkbox("Quick run (faster training)", value=True)

# Hyperparameters (small UI)
st.sidebar.markdown("LSTM & training")
look_back = st.sidebar.slider("LSTM look_back", 1, 120, 60)
train_frac = st.sidebar.slider("Train fraction", 0.5, 0.95, 0.8, step=0.01)

st.sidebar.markdown("---")
st.sidebar.write("Extras")
download_all = st.sidebar.checkbox("Enable CSV downloads", True)

# ---------------------------
# Utility functions
# ---------------------------
def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(('.xlsx','.xls')):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unknown file format")

def infer_time_and_rate_cols(df):
    # time candidate
    time_candidates = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    time_col = time_candidates[0] if time_candidates else df.columns[0]
    # rate candidates (common currency keywords)
    rate_candidates = [c for c in df.columns if any(x in c.upper() for x in ['EURO','EUR','USD','GBP','JPY','RATE','EXCHANGE'])]
    # If multi-currency, pick first numeric col not time
    if len(rate_candidates) == 0:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        rate_col = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns)>1 else df.columns[0]
    else:
        rate_col = rate_candidates[0]
    return time_col, rate_col

def fetch_live_rates(base, symbols, days):
    """
    Fetch historical rates from exchangerate.host.
    Returns DataFrame indexed by date with columns for each symbol.
    """
    symbols = [s.strip().upper() for s in symbols.split(",")]
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    url = f"https://api.exchangerate.host/timeseries"
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "base": base,
        "symbols": ",".join(symbols)
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError("Failed fetching rates: " + str(data))
    rates = data["rates"]
    rows = []
    for d, vals in rates.items():
        row = {"date": pd.to_datetime(d)}
        row.update(vals)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df

def create_features(df):
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    for lag in range(1,8):
        df[f'lag_{lag}'] = df['Rate'].shift(lag)
    df['roll_mean_7'] = df['Rate'].rolling(7).mean()
    df['roll_std_7'] = df['Rate'].rolling(7).std()
    df = df.dropna()
    return df

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def evaluate(y_true, y_pred):
    dfc = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    if dfc.empty:
        return {'RMSE':np.nan, 'MAE':np.nan, 'MAPE':np.nan}
    return {
        'RMSE': np.sqrt(mean_squared_error(dfc['y_true'], dfc['y_pred'])),
        'MAE': mean_absolute_error(dfc['y_true'], dfc['y_pred']),
        'MAPE': mape(dfc['y_true'], dfc['y_pred'])
    }

def make_sequences(series_vals, look_back):
    scaler = MinMaxScaler((0,1)).fit(series_vals.reshape(-1,1))
    scaled = scaler.transform(series_vals.reshape(-1,1))
    X,y = [],[]
    for i in range(len(scaled)-look_back):
        X.append(scaled[i:i+look_back,0])
        y.append(scaled[i+look_back,0])
    X = np.array(X).reshape(-1, look_back, 1)
    return X, np.array(y), scaler

def df_to_csv_bytes(df):
    b = io.BytesIO()
    df.to_csv(b, index=True)
    b.seek(0)
    return b

# ---------------------------
# Load data according to selection
# ---------------------------
st.title("📊 FX Forecast Dashboard — Full (Multi-Currency + Live + AI Insights)")
st.markdown(f"**Theme:** {theme_choice} — use the sidebar to change settings and data source.")

# Load data
df_main = None
currency_columns = []
selected_currency = None

try:
    if data_mode == "Upload Excel/CSV (multi-currency)":
        if uploaded is None:
            st.info("Upload a file to continue.")
            st.stop()
        raw = read_uploaded_file(uploaded)
        time_col, rate_col = infer_time_and_rate_cols(raw)
        # If dataframe has multiple numeric currency columns, show them for user to pick
        numeric_cols = raw.select_dtypes(include='number').columns.tolist()
        # if numeric columns include many, assume multi-currency file where column names are currency rates
        if len(numeric_cols) >= 2:
            currency_columns = numeric_cols
        else:
            # try to use header names that look like currency labels
            currency_columns = [c for c in raw.columns if any(x in c.upper() for x in ['EUR','USD','GBP','JPY','AUD','CAD','CHF','CNY','INR'])]
            if not currency_columns:
                # fallback to any non-time columns
                currency_columns = [c for c in raw.columns if c != time_col][:5]
        # pick currency to analyze (default first)
        selected_currency = st.sidebar.selectbox("Pick currency column to analyze", options=currency_columns, index=0)
        df_main = raw[[time_col, selected_currency]].rename(columns={time_col:'Time', selected_currency:'Rate'})
        df_main['Time'] = pd.to_datetime(df_main['Time'], errors='coerce')
        df_main = df_main.set_index('Time').sort_index()
        df_main['Rate'] = pd.to_numeric(df_main['Rate'], errors='coerce')
        df_main['Rate'] = df_main['Rate'].ffill().bfill()
    elif data_mode == "Use sample file in folder":
        raw = pd.read_excel("Foreign_Exchange_Rates[1].xlsx")
        time_col, rate_col = infer_time_and_rate_cols(raw)
        # allow user to pick among numeric columns
        numeric_cols = raw.select_dtypes(include='number').columns.tolist()
        currency_columns = numeric_cols if numeric_cols else [rate_col]
        selected_currency = st.sidebar.selectbox("Pick currency column to analyze", options=currency_columns, index=0)
        df_main = raw[[time_col, selected_currency]].rename(columns={time_col:'Time', selected_currency:'Rate'})
        df_main['Time'] = pd.to_datetime(df_main['Time'], errors='coerce')
        df_main = df_main.set_index('Time').sort_index()
        df_main['Rate'] = pd.to_numeric(df_main['Rate'], errors='coerce')
        df_main['Rate'] = df_main['Rate'].ffill().bfill()
    else:
        # Live mode
        df_live = fetch_live_rates(base_currency, symbols_input, live_days)
        # user picks one symbol to analyze
        cols = df_live.columns.tolist()
        selected_currency = st.sidebar.selectbox("Pick currency to analyze (fetched)", options=cols, index=0)
        df_main = df_live[[selected_currency]].rename(columns={selected_currency:'Rate'})
        df_main.index = pd.to_datetime(df_main.index)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if df_main is None or df_main['Rate'].isna().all():
    st.error("No valid rate series found. Check your input file or live API response.")
    st.stop()

# Basic series info cards
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    st.subheader(f"Analyzing: {selected_currency}")
    st.dataframe(df_main.head(8))
with col2:
    st.metric("Observations", f"{len(df_main):,}")
with col3:
    st.metric("Start", str(df_main.index.min().date()))
with col4:
    st.metric("End", str(df_main.index.max().date()))

# ---------------------------
# EDA & Visualization
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview & EDA", "Models & Metrics", "Forecasts", "AI Insights"])

with tab1:
    st.markdown("### Time series overview")
    fig = px.line(df_main, y='Rate', title=f"{selected_currency} — Rate over time", labels={'index':'Time','Rate':'Rate'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribution & Summary")
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df_main.reset_index(), x='Rate', nbins=80, title="Distribution of Rates")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.write(df_main['Rate'].describe().to_frame().T)

    # seasonality (monthly)
    if pd.infer_freq(df_main.index) is not None:
        monthly = df_main['Rate'].resample('M').mean()
        fig_month = px.line(monthly, title="Monthly mean", labels={'index':'Time', 'value':'Mean Rate'})
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("Unknown frequency: monthly resample omitted.")

# ---------------------------
# Prepare ML datasets
# ---------------------------
df_feat = create_features(df_main)
train_n = int(len(df_main) * train_frac)
train = df_main.iloc[:train_n]
test = df_main.iloc[train_n:]

# For ML models
train_ml_n = int(len(df_feat) * train_frac)
train_ml = df_feat.iloc[:train_ml_n]
test_ml = df_feat.iloc[train_ml_n:]

# LSTM sequences
X_seq, y_seq, seq_scaler = make_sequences = None, None, None
try:
    X_seq, y_seq, seq_scaler = make_sequences or None, None, None
except:
    pass

# Build LSTM sequences only if needed
if use_lstm:
    try:
        X_seq, y_seq, seq_scaler = make_sequences = None, None, None
        # use helper defined earlier
        X_seq, y_seq, seq_scaler = make_sequences = None, None, None
    except:
        pass

# NOTE: we actually create sequences in the training part below to avoid distraction if LSTM disabled.

# ---------------------------
# Model training (when user clicks)
# ---------------------------
run_flag = st.button("Train & Evaluate Models")

if run_flag:
    with st.spinner("Training models — this may take a while depending on options..."):
        metrics = {}
        preds = {}

        # ARIMA
        if use_arima:
            try:
                ar = ARIMA(train['Rate'], order=(5,1,0)).fit()
                ar_pred = pd.Series(ar.forecast(steps=len(test)), index=test.index[:len(test)])
                metrics['ARIMA'] = evaluate(test['Rate'].loc[ar_pred.index], ar_pred)
                preds['ARIMA'] = ar_pred
            except Exception as e:
                st.warning(f"ARIMA failed: {e}")
                metrics['ARIMA'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                preds['ARIMA'] = pd.Series(dtype=float)

        # SARIMA
        if use_sarima:
            try:
                sar = SARIMAX(train['Rate'], order=(1,1,1), seasonal_order=(1,1,1,30),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                sar_pred = pd.Series(sar.forecast(steps=len(test)), index=test.index[:len(test)])
                metrics['SARIMA'] = evaluate(test['Rate'].loc[sar_pred.index], sar_pred)
                preds['SARIMA'] = sar_pred
            except Exception as e:
                st.warning(f"SARIMA failed: {e}")
                metrics['SARIMA'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                preds['SARIMA'] = pd.Series(dtype=float)

        # Prophet
        if use_prophet:
            try:
                ptrain = train.reset_index().rename(columns={train.index.name or 'index':'ds','Rate':'y'})
                if 'ds' not in ptrain.columns:
                    ptrain = ptrain.rename(columns={ptrain.columns[0]:'ds'})
                m = Prophet()
                m.fit(ptrain[['ds','y']].dropna())
                future = m.make_future_dataframe(periods=len(test))
                fc = m.predict(future).set_index('ds')['yhat'].reindex(test.index).fillna(method='ffill').fillna(method='bfill')[:len(test)]
                metrics['Prophet'] = evaluate(test['Rate'].loc[fc.dropna().index], fc.dropna())
                preds['Prophet'] = fc
            except Exception as e:
                st.warning(f"Prophet failed: {e}")
                metrics['Prophet'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                preds['Prophet'] = pd.Series(dtype=float)

        # XGBoost
        if use_xgb:
            try:
                feats = [c for c in train_ml.columns if c not in ['date','Rate']]
                xg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100 if quick_run else 500, learning_rate=0.01, n_jobs=-1)
                xg.fit(train_ml[feats], train_ml['Rate'])
                xg_pred = pd.Series(xg.predict(test_ml[feats]), index=test_ml.index)
                metrics['XGBoost'] = evaluate(test_ml['Rate'], xg_pred)
                preds['XGBoost'] = xg_pred
            except Exception as e:
                st.warning(f"XGBoost failed: {e}")
                metrics['XGBoost'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                preds['XGBoost'] = pd.Series(dtype=float)

        # LSTM
        if use_lstm:
            try:
                # create sequences now
                X_all, y_all, scaler_seq = make_sequences(np.array(df_main['Rate'].values), look_back)
                train_seq_n = train_n - look_back
                if train_seq_n <= 0:
                    st.warning("Train size too small for LSTM with chosen look_back. Skipping LSTM.")
                    metrics['LSTM'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                    preds['LSTM'] = pd.Series(dtype=float)
                else:
                    X_train_seq, X_test_seq = X_all[:train_seq_n], X_all[train_seq_n:]
                    y_train_seq, y_test_seq = y_all[:train_seq_n], y_all[train_seq_n:]
                    model = Sequential()
                    model.add(LSTM(50, input_shape=(X_train_seq.shape[1], 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_train_seq, y_train_seq, epochs=10 if quick_run else 50, batch_size=32, verbose=0)
                    p_scaled = model.predict(X_test_seq)
                    p = scaler_seq.inverse_transform(p_scaled).flatten()
                    idx = test.index[:len(p)]
                    p_series = pd.Series(p, index=idx)
                    y_actual = scaler_seq.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
                    metrics['LSTM'] = evaluate(pd.Series(y_actual, index=idx), p_series)
                    preds['LSTM'] = p_series
            except Exception as e:
                st.warning(f"LSTM failed: {e}")
                metrics['LSTM'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                preds['LSTM'] = pd.Series(dtype=float)

        # CatBoost
        if use_cat:
            try:
                feats = [c for c in train_ml.columns if c not in ['date','Rate']]
                cb = CatBoostRegressor(iterations=50 if quick_run else 300, learning_rate=0.01, depth=6, verbose=0, random_seed=42)
                cb.fit(train_ml[feats], train_ml['Rate'], eval_set=(test_ml[feats], test_ml['Rate']), early_stopping_rounds=20, use_best_model=True)
                cb_pred = pd.Series(cb.predict(test_ml[feats]), index=test_ml.index)
                metrics['CatBoost'] = evaluate(test_ml['Rate'], cb_pred)
                preds['CatBoost'] = cb_pred
            except Exception as e:
                st.warning(f"CatBoost failed: {e}")
                metrics['CatBoost'] = {'RMSE':np.nan,'MAE':np.nan,'MAPE':np.nan}
                preds['CatBoost'] = pd.Series(dtype=float)

        # Display results in the Models & Metrics tab
        with tab2:
            st.subheader("Model Performance Summary")
            perf_df = pd.DataFrame.from_dict(metrics, orient='index')
            st.dataframe(perf_df.style.format({"RMSE":"{:.6f}","MAE":"{:.6f}","MAPE":"{:.2f}%"}), use_container_width=True)

            # leaderboard
            if not perf_df.empty:
                perf_sorted = perf_df.sort_values('RMSE')
                fig_lead = px.bar(perf_sorted.reset_index().rename(columns={'index':'Model'}), x='Model', y='RMSE', title="Models by RMSE")
                st.plotly_chart(fig_lead, use_container_width=True)

            if download_all:
                st.download_button("Download metrics CSV", data=df_to_csv_bytes(perf_df), file_name="fx_model_metrics.csv")

        # Forecasts tab
        with tab3:
            st.subheader("Forecasts vs Actuals")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test.index, y=test['Rate'], mode='lines', name='Actual', line=dict(color='black', width=3)))
            for name, ser in preds.items():
                if ser is not None and not ser.empty:
                    fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode='lines', name=name, line=dict(dash='dash')))
            fig.update_layout(title=f"Forecasts vs Actual for {selected_currency}", xaxis_title="Time", yaxis_title="Rate", legend=dict(orientation='h'))
            st.plotly_chart(fig, use_container_width=True)

            # combined table
            combined = pd.DataFrame({'Actual': test['Rate']})
            for k,v in preds.items():
                combined[k] = v
            st.dataframe(combined.tail(30), use_container_width=True)
            if download_all:
                st.download_button("Download forecasts CSV", data=df_to_csv_bytes(combined), file_name="fx_forecasts.csv")

        # AI Insights tab
        with tab4:
            st.subheader("Automated AI Insights")
            insights = []
            recent = df_main['Rate'].iloc[-90:] if len(df_main)>=90 else df_main['Rate']
            # trend
            try:
                slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
            except:
                slope = 0
            if slope > 0:
                insights.append(f"Upward short-term trend detected (slope={slope:.6f}).")
            elif slope < 0:
                insights.append(f"Downward short-term trend detected (slope={slope:.6f}).")
            else:
                insights.append("No clear short-term trend detected.")

            # volatility
            vol = recent.pct_change().std() * np.sqrt(252) if len(recent)>1 else np.nan
            insights.append(f"Volatility (annualized approx): {vol:.4f}")

            # recent change
            change_pct = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100 if recent.iloc[0] != 0 else np.nan
            insights.append(f"Change over last {len(recent)} points: {change_pct:.2f}%")

            # best model
            perf_clean = {k:v for k,v in metrics.items() if v.get('RMSE')==v.get('RMSE')}
            if perf_clean:
                best = min(perf_clean.items(), key=lambda x: x[1]['RMSE'])
                insights.append(f"Best model by RMSE: {best[0]} (RMSE={best[1]['RMSE']:.6f}).")
            else:
                insights.append("Insufficient metric data to recommend best model.")

            # anomalies
            z = (df_main['Rate'] - df_main['Rate'].rolling(30).mean())/df_main['Rate'].rolling(30).std()
            recent_anoms = z[np.abs(z)>3].tail(5)
            if not recent_anoms.empty:
                insights.append("Recent anomalies detected at: " + ", ".join(str(i.date()) for i in recent_anoms.index))
            else:
                insights.append("No strong recent anomalies (z-score>3).")

            for it in insights:
                st.markdown("- " + it)

            summary_text = f"In the recent window the series moved {change_pct:.2f}% with trend {'up' if slope>0 else 'down' if slope<0 else 'flat'}. Best model: {best[0] if perf_clean else 'N/A'}."
            st.info(summary_text)

        st.success("Training & evaluation complete.")
else:
    st.info("Press **Train & Evaluate Models** to start training the selected models.")

# Footer
st.markdown("---")
st.caption("FX Forecast Dashboard — Theme and layout are cosmetic. AI Insights are deterministic heuristics (no LLM). Live rates come from exchangerate.host.")
