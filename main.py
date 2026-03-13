from data.synthetic_data import load_synthetic_data 
from features.feature_engineering import prepare_features 
from models.risk_classifier import train_models 
from models.anomaly_detector import detect_anomalies
from drift.drift_detector import detect_risk_drift
from visualization.plots import plot_risk_distribution
from visualization.plots import plot_anomalies 
from visualization.plots import plot_risk_scatter 
from visualization.plots import plot_dominant_asset_distribution
from visualization.plots import plot_feature_importance 

def main():
    df=load_synthetic_data()
    print("Original Data:")
    print(df.head())


    features,target=prepare_features(df)
    print("\nFeatures:") 
    print(features.head()) 
    print("\nTarget:") 
    print(target.unique())
    print(target.head()) 


    rf_model, xgb_model=train_models(features,target) 

    
    anomalies = detect_anomalies(features )
    df["anomaly"] = anomalies
    print("\nAnomaly Detection Results:")
    print(df["anomaly"].value_counts())
    print("\nSample Anomalous Portfolios:")
    print(df[df["anomaly"] == -1].head())


    df = detect_risk_drift(df)
    print("\nRisk Drift Summary:")
    print(df["risk_drift"].value_counts())
    print("\nSample Drift Cases:")
    print(df[df["risk_drift"] == True][[
    "client_id",
    "month",
    "volatility_change",
    "concentration_change",
    "equity_change",
    "drift_reason"
]].head())


    plot_risk_distribution(df) 
    plot_anomalies(df) 
    plot_risk_scatter(df) 
    plot_dominant_asset_distribution(df) 
    plot_feature_importance(rf_model, features.columns) 
    plot_feature_importance(rf_model, features.columns)




if __name__ == "__main__": 
    main() 