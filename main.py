from data.user_input import get_user_portfolio
from data.market_data_fetcher import MarketDataFetcher
from features.feature_engineering import prepare_features
from models.risk_classifier import train_models
from models.anomaly_detector import detect_anomalies
from drift.drift_detector import detect_risk_drift
from visualization.plots import plot_risk_distribution, plot_anomalies, plot_risk_scatter, plot_feature_importance 
import pandas as pd

def process_single_asset(ticker: str, quantity: float, purchase_date: str):
    print(f"\n=========================================")
    print(f"PROCESSING ASSET: {ticker}")
    print(f"=========================================")
    
    fetcher = MarketDataFetcher()
    
    # ---------------------------------------------------------
    # 1. THE HISTORY (TRAINING PHASE)
    # ---------------------------------------------------------
    df_history_raw = fetcher.get_full_history(ticker)
    df_history, features_train, target_train = prepare_features(df_history_raw, is_training=True)
    
    print(f"\nTraining ML Models on {len(features_train)} historical days of {ticker}...")
    rf_model, xgb_model = train_models(features_train, target_train)
    
    # ---------------------------------------------------------
    # 2. THE SHOWCASE (EVALUATION PHASE)
    # ---------------------------------------------------------
    df_eval_raw = fetcher.get_evaluation_window(ticker, purchase_date)
    df_eval, features_eval, _ = prepare_features(df_eval_raw, is_training=False)
    
    print(f"\nEvaluating User Holdings ({len(features_eval)} days since {purchase_date})...")
    
    # Calculate real Portfolio Value
    df_eval["portfolio_value"] = df_eval["price"] * quantity
    
    # ---------------------------------------------------------
    # 3. PREDICTIONS & DRIFT
    # ---------------------------------------------------------
    # A. Predict Risk Level (using XGBoost for example)
    risk_predictions = xgb_model.predict(features_eval)
    
    # Map numerical predictions back to labels for display
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    df_eval["predicted_risk"] = [risk_map[pred] for pred in risk_predictions]
    
    # B. Anomaly Detection
    anomalies = detect_anomalies(features_eval)
    df_eval["is_anomaly"] = anomalies == -1 # Returns -1 for anomalies
    
    # C. Drift Detection
    df_eval = detect_risk_drift(df_eval)
    
    # ---------------------------------------------------------
    # 4. SUMMARY
    # ---------------------------------------------------------
    print("\n--- Summary for User's Holding Period ---")
    print("Risk Levels Detected:")
    print(df_eval["predicted_risk"].value_counts())
    
    print("\nDays flagged as Anomalous behavior:")
    print(df_eval["is_anomaly"].sum())
    
    print("\nDays flagged for Risk Drift:")
    print(df_eval["risk_drift"].sum())
    if df_eval["risk_drift"].sum() > 0:
        print("\nSample Drift Explanations:")
        print(df_eval[df_eval["risk_drift"]][['Date', 'drift_reason']].head())

    # Visualizations
    # Note: Using try/except in case running on a server without display
    try:
        plot_risk_distribution(df_eval)
        plot_anomalies(df_eval)
        plot_risk_scatter(df_eval)
        plot_feature_importance(xgb_model, features_eval.columns)
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

    # Return the evaluated dataframe if needed for a frontend UI
    return df_eval

def main():
    portfolio = get_user_portfolio()
    
    results = {}
    for asset in portfolio:
        df_result = process_single_asset(
            ticker=asset["ticker"], 
            quantity=asset["quantity"], 
            purchase_date=asset["purchase_date"]
        )
        results[asset["ticker"]] = df_result
        
    print("\nAll assets processed successfully!")

if __name__ == "__main__":
    main() 