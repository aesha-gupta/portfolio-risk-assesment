import streamlit as st
import pandas as pd
import datetime
from data.market_data_fetcher import MarketDataFetcher
from features.feature_engineering import prepare_features
from models.risk_classifier import train_models
from models.anomaly_detector import detect_anomalies
from drift.drift_detector import detect_risk_drift
from visualization.streamlit_plots import (
    plot_risk_over_time, 
    plot_risk_scatter_interactive,
    plot_anomalies_interactive,
    plot_feature_importance_interactive
)

st.set_page_config(page_title="Portfolio Risk ML", layout="wide", page_icon="📈")

st.title("📈 Machine Learning Portfolio Risk Assessment")
st.markdown("Enter your exact stock holding below. The system will pull up to 20 years of real market data, train customized ML models on your specific asset's history, and evaluate your personal holding period for hidden risks and anomalies.")

st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="e.g. AAPL, MSFT, TSLA")
quantity = st.sidebar.number_input("Quantity Held", min_value=1.0, value=50.0)

# Streamlit uses separate widgets for date and time.
purchase_date = st.sidebar.date_input("Purchase Date", value=datetime.date(2023, 1, 15))
purchase_time = st.sidebar.time_input("Purchase Time", value=datetime.time(10, 30))

run_button = st.sidebar.button("Run ML Pipeline", type="primary")

if run_button:
    # Format the date back into the string our pipeline expects
    full_datetime_str = f"{purchase_date.strftime('%Y-%m-%d')} {purchase_time.strftime('%H:%M:%S')}"
    
    st.divider()
    st.subheader(f"Analyzing {quantity} shares of {ticker} since {full_datetime_str}")
    
    fetcher = MarketDataFetcher()
    
    # 1. Training Phase
    with st.status("🧠 Step 1: Training the Machine Learning Engine...", expanded=True) as status:
        st.write(f"Downloading 20-year history for {ticker} via yfinance...")
        df_history_raw = fetcher.get_full_history(ticker)
        
        st.write("Engineering quantitative features (Volatility, Drawdowns, Momentum)...")
        df_history, features_train, target_train = prepare_features(df_history_raw, is_training=True)
        
        st.write(f"Training Random Forest and XGBoost models on {len(features_train)} historical days...")
        rf_model, xgb_model = train_models(features_train, target_train)
        
        status.update(label="Step 1 Complete: Models Trained Successfully!", state="complete", expanded=False)

    # 2. Evaluation Phase
    with st.status("🔍 Step 2: Evaluating Your Specific Portfolio...", expanded=True) as status:
        st.write(f"Downloading market data since your purchase on {full_datetime_str}...")
        df_eval_raw = fetcher.get_evaluation_window(ticker, full_datetime_str)
        
        st.write("Extracting your specific engineered features...")
        df_eval, features_eval, _ = prepare_features(df_eval_raw, is_training=False)
        
        if len(features_eval) == 0:
            status.update(label="Error: Not enough data.", state="error")
            st.error("Not enough trading days since your purchase to generate rolling 90-day features. Try an older date.")
            st.stop()
            
        st.write("Calculating Portfolio Value...")
        df_eval["portfolio_value"] = df_eval["price"] * quantity
        
        st.write("Running Risk Classifiers...")
        risk_predictions = xgb_model.predict(features_eval)
        risk_map = {0: "Low", 1: "Medium", 2: "High"}
        df_eval["predicted_risk"] = [risk_map[pred] for pred in risk_predictions]
        
        st.write("Detecting Anomalies using Isolation Forest...")
        anomalies = detect_anomalies(features_eval)
        df_eval["is_anomaly"] = anomalies == -1
        
        st.write("Scanning for abrupt Risk Drift...")
        df_eval = detect_risk_drift(df_eval)
        
        status.update(label="Step 2 Complete: Portfolio Evaluated!", state="complete", expanded=False)

    # 3. Visualization Phase
    st.divider()
    st.subheader("📊 Assessment Results")
    
    # Top Line Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Days Analyzed", len(df_eval))
    with col2:
        current_sharpe = df_eval["sharpe_ratio_30d"].iloc[-1]
        st.metric("Current Sharpe Ratio", round(current_sharpe, 2))
    with col3:
        high_risk_days = (df_eval["predicted_risk"] == "High").sum()
        st.metric("High-Risk Days", high_risk_days, str(round(high_risk_days/len(df_eval)*100, 1)) + "%")
    with col4:
        anomalies_found = df_eval["is_anomaly"].sum()
        st.metric("Anomalies Detected", anomalies_found, delta_color="inverse")
    with col5:
        drifts = df_eval["risk_drift"].sum()
        st.metric("Risk Drifts Found", drifts, delta_color="inverse")

    st.markdown("### Risk over Time")
    st.plotly_chart(plot_risk_over_time(df_eval), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Volatility vs Drawdown Matrix")
        st.plotly_chart(plot_risk_scatter_interactive(df_eval), use_container_width=True)
    with col2:
        st.markdown("### Anomaly Detection Map")
        st.plotly_chart(plot_anomalies_interactive(df_eval), use_container_width=True)
        
    st.markdown("### Feature Importance Matrix")
    st.plotly_chart(plot_feature_importance_interactive(xgb_model, features_eval.columns), use_container_width=True)
    
    # Show Raw Data if drift is found
    if drifts > 0:
        st.warning(f"Warning: {drifts} major structural drift events were detected in your portfolio's risk profile.")
        st.dataframe(df_eval[df_eval["risk_drift"]][['Date', 'drift_reason', 'price', 'predicted_risk']], use_container_width=True)
        
    with st.expander("View Raw DataFrame"):
        st.dataframe(df_eval.tail(100))
