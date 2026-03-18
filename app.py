import streamlit as st
import pandas as pd
import datetime
import numpy as np
import shap
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

st.title("📈 Multi-Asset Portfolio Risk Assessment")
st.markdown("Enter your exact stock holdings below. The system will evaluate them individually AND aggregate them into a single holistic portfolio analysis.")

st.sidebar.header("Your Portfolio Builder")
st.sidebar.markdown("Add as many assets as you want. Provide the exact purchase dates to evaluate your actual holding period.")

# Initialize default portfolio
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(
        [
            {"Ticker": "AAPL", "Quantity": 50.0, "Date": datetime.date(2023, 1, 15)},
            {"Ticker": "MSFT", "Quantity": 20.0, "Date": datetime.date(2023, 1, 15)}
        ]
    )

st.sidebar.markdown("---")

edited_portfolio = st.sidebar.data_editor(
    st.session_state.portfolio_df, 
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

run_button = st.sidebar.button("Run Portfolio ML Pipeline", type="primary")

if run_button:
    if edited_portfolio.empty or edited_portfolio["Ticker"].isna().all():
        st.error("Please add at least one valid ticker to your portfolio.")
        st.stop()
        
    st.divider()
    fetcher = MarketDataFetcher()
    
    all_features_train = []
    all_targets_train = []
    individual_evals = {}
    
    st.subheader("Step 1: Processing Individual Assets")
    with st.expander("Show Detailed Asset Extraction Logs", expanded=True):
        for index, row in edited_portfolio.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            if not ticker or ticker == "NAN": continue
            
            quantity = float(row["Quantity"])
            p_date = row["Date"]
            full_datetime_str = f"{p_date.strftime('%Y-%m-%d')} 00:00:00"
            
            st.markdown(f"**Fetching `{ticker}`...**")
            
            df_history_raw = fetcher.get_full_history(ticker)
            if df_history_raw.empty: continue
                
            df_history, f_train, t_train = prepare_features(df_history_raw, is_training=True)
            all_features_train.append(f_train)
            all_targets_train.append(t_train)
            
            df_eval_raw = fetcher.get_evaluation_window(ticker, full_datetime_str)
            if df_eval_raw.empty: continue
                
            df_eval, features_eval, _ = prepare_features(df_eval_raw, is_training=False)
            if len(features_eval) == 0: continue
            
            df_eval["portfolio_value"] = df_eval["price"] * quantity
            df_eval["ticker"] = ticker
            individual_evals[ticker] = df_eval
            st.write(f"- ✔ {ticker} processed ({len(df_eval)} evaluation days).")

    if not individual_evals:
        st.error("No valid assets were processed. Cannot build portfolio.")
        st.stop()

    st.subheader("Step 2: Training Unified Risk Models")
    with st.spinner("Training models on historical data from all your assets combined..."):
        combined_features = pd.concat(all_features_train, ignore_index=True)
        combined_targets = pd.concat(all_targets_train, ignore_index=True)
        rf_model, xgb_model = train_models(combined_features, combined_targets)
        st.success("Unified Models Trained Successfully!")

    st.subheader("Step 3: Aggregating Portfolio")
    with st.spinner("Aligning timelines and extracting aggregate features..."):
        value_dfs = []
        for ticker, df in individual_evals.items():
            val_df = df[["Date", "portfolio_value"]].rename(columns={"portfolio_value": ticker})
            value_dfs.append(val_df)
            
        import functools
        master_df = functools.reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), value_dfs)
        master_df = master_df.sort_values("Date").reset_index(drop=True)
        master_df.fillna(method="ffill", inplace=True)
        master_df.fillna(0, inplace=True)
        
        ticker_cols = list(individual_evals.keys())
        master_df["total_portfolio_value"] = master_df[ticker_cols].sum(axis=1)
        master_df["largest_asset_value"] = master_df[ticker_cols].max(axis=1)
        master_df["concentration_pct"] = master_df["largest_asset_value"] / master_df["total_portfolio_value"]
        
        portfolio_price_df = pd.DataFrame({
            "Date": master_df["Date"],
            "price": master_df["total_portfolio_value"],
            "ticker": "TOTAL PORTFOLIO"
        })
        
        port_eval, port_features, _ = prepare_features(portfolio_price_df, is_training=False)
        port_eval = pd.merge(port_eval, master_df[["Date", "concentration_pct", "total_portfolio_value"]], on="Date", how="left")
        port_eval["portfolio_value"] = port_eval["total_portfolio_value"] 
        
        st.write("Running Risk Classifiers on Aggregate Portfolio...")
        risk_predictions = xgb_model.predict(port_features)
        risk_map = {0: "Low", 1: "Medium", 2: "High"}
        port_eval["predicted_risk"] = [risk_map[pred] for pred in risk_predictions]
        
        anomalies, anomaly_scores = detect_anomalies(port_features, return_scores=True)
        port_eval["is_anomaly"] = anomalies == -1
        port_eval["anomaly_score"] = anomaly_scores
        
        port_eval = detect_risk_drift(port_eval)
        st.success("Portfolio Aggregation & Prediction Complete!")

    # -------------------------------------------------------------
    # 5. FINANCE + ML HYBRID DASHBOARD
    # -------------------------------------------------------------
    st.divider()
    st.title("🎯 Ideal Output (Finance + ML Combined)")
    
    # Pre-calculate metrics for the dashboard
    current_features = port_features.iloc[[-1]]
    current_risk_prob = xgb_model.predict_proba(current_features)[0]
    high_risk_prob = current_risk_prob[2] # Class 2 is High Risk
    risk_score = int(high_risk_prob * 100)
    current_risk_label = port_eval["predicted_risk"].iloc[-1]
    
    current_anomaly = port_eval["is_anomaly"].iloc[-1]
    current_anomaly_score = port_eval["anomaly_score"].iloc[-1]
    current_concentration = port_eval["concentration_pct"].iloc[-1]
    mean_return_ann = port_eval["daily_return"].mean() * 252
    
    # ✅ 1. Final Risk Summary
    st.markdown("### ✅ 1. Final Risk Summary")
    status_color = "🔴" if current_risk_label == "High" else ("🟠" if current_risk_label == "Medium" else "🟢")
    anomaly_flag = "YES ⚠️ (Unusual behavior detected)" if current_anomaly else "NO ✅"
    
    st.markdown(f"**Portfolio Risk Level:** {current_risk_label.upper()} {status_color}")
    st.markdown(f"**Risk Score:** {risk_score} / 100")
    st.markdown(f"**Anomaly Status:** {anomaly_flag}")
    
    # ✅ 2. Key Risk Metrics
    st.markdown("---")
    st.markdown("### ✅ 2. Key Risk Metrics (Quant Evidence)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Expected Return", f"{mean_return_ann*100:.1f}%")
    col2.metric("Volatility", f"{port_eval['volatility_30d'].iloc[-1]*100:.1f}%")
    col3.metric("Sharpe Ratio", f"{port_eval['sharpe_ratio_30d'].iloc[-1]:.2f}")
    col4.metric("Max Drawdown", f"{port_eval['max_drawdown_90d'].iloc[-1]*100:.1f}%")
    col5.metric("Diversification Score", "Low" if current_concentration > 0.5 else "High")
    
    # ✅ 3. Model Explanation
    st.markdown("---")
    st.markdown("### ✅ 3. Model Explanation")
    
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(port_features)
        # XGBoost multi:softprob returns a list of shap arrays per class. High risk is index 2.
        if isinstance(shap_values, list):
            latest_shap_high_risk = shap_values[2][-1] 
        else:
            latest_shap_high_risk = shap_values[-1]
            
        feature_contributions = pd.DataFrame({
            "Feature": port_features.columns,
            "Contribution": latest_shap_high_risk
        }).sort_values(by="Contribution", key=abs, ascending=False)
        
        st.markdown("**Top Factors Contributing to Risk:**")
        for _, row in feature_contributions.head(3).iterrows():
            feat = row["Feature"].replace("_", " ").title()
            val = row["Contribution"]
            sign = "+" if val > 0 else "-"
            st.markdown(f"**{sign} High {feat}** ({sign}{abs(val):.2f})")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

    # ✅ 4. Model Confidence / Probability
    st.markdown("---")
    st.markdown("### ✅ 4. Model Confidence / Probability")
    st.markdown(f"**Probability of High Risk:** {high_risk_prob*100:.1f}%")
    st.markdown("**Model Used:** XGBoost (Best performing)")
    
    # ✅ 5. Anomaly Detection Detail
    st.markdown("---")
    st.markdown("### ✅ 5. Anomaly Detection (Isolation Forest Output)")
    st.markdown(f"**Anomaly Score:** {current_anomaly_score:.2f}")
    st.markdown("**Status:** " + ("Anomalous Portfolio" if current_anomaly else "Normal Portfolio"))
    if current_anomaly:
        st.markdown("**Reason:**")
        if current_concentration > 0.6:
            st.markdown("- Extremely high concentration in top assets")
        if port_eval['volatility_30d'].iloc[-1] > 0.03:
            st.markdown("- Unusual volatility spike detected")
            
    # ✅ 6. Actionable Insights
    st.markdown("---")
    st.markdown("### ✅ 6. Actionable Insights")
    st.markdown("**Recommendations:**")
    if current_concentration > 0.5:
        st.markdown(f"- Reduce exposure to your top asset (currently {current_concentration*100:.1f}% of portfolio)")
    if port_eval['volatility_30d'].iloc[-1] > 0.02:
        st.markdown("- Add low-volatility assets (e.g., index funds)")
    if current_concentration > 0.5:
        st.markdown("- Improve diversification across sectors to hedge against targeted drawdowns.")
    if current_concentration <= 0.5 and port_eval['volatility_30d'].iloc[-1] <= 0.02:
        st.markdown("- Portfolio is well-balanced and stable. No immediate structural changes recommended.")

    # Show the gorgeous sub-plots so they don't lose the old features
    st.divider()
    st.markdown("### Portfolio Trajectory & Risk Map")
    st.plotly_chart(plot_risk_over_time(port_eval), use_container_width=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(plot_risk_scatter_interactive(port_eval), use_container_width=True)
    with colB:
        st.plotly_chart(plot_anomalies_interactive(port_eval), use_container_width=True)
        
    # -------------------------------------------------------------
    # 6. INDIVIDUAL ASSET BREAKDOWNS (TABS)
    # -------------------------------------------------------------
    st.divider()
    st.header("🔍 Individual Asset Breakdowns")
    tabs = st.tabs(list(individual_evals.keys()))
    
    for idx, (ticker, df) in enumerate(individual_evals.items()):
        with tabs[idx]:
            df_features = df[["daily_return", "volatility_30d", "momentum_30d", "max_drawdown_90d", "sharpe_ratio_30d"]]
            df["predicted_risk"] = [risk_map[p] for p in xgb_model.predict(df_features)]
            df["is_anomaly"] = detect_anomalies(df_features, return_scores=False) == -1
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{ticker} Days", len(df))
            c2.metric("Sharpe Ratio", round(df["sharpe_ratio_30d"].iloc[-1], 2))
            c3.metric("High-Risk Days", (df["predicted_risk"] == "High").sum())
            c4.metric("Anomalies", df["is_anomaly"].sum())

            st.plotly_chart(plot_risk_over_time(df), use_container_width=True)
