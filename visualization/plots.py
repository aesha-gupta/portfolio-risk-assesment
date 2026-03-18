import matplotlib.pyplot as plt
import pandas as pd

def plot_risk_distribution(df):
    counts = df["predicted_risk"].value_counts()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Predicted Portfolio Risk Distribution over Time")
    plt.xlabel("Risk Level")
    plt.ylabel("Days in State")
    plt.show()

def plot_anomalies(df):
    counts = df["is_anomaly"].value_counts()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Anomaly Detection Results (User Timeframe)")
    plt.xlabel("Is Anomaly")
    plt.ylabel("Days")
    plt.show()

def plot_risk_scatter(df):
    colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }
    
    plt.figure(figsize=(10, 6))

    for risk in df["predicted_risk"].unique():
        subset = df[df["predicted_risk"] == risk]
        
        # Plot Volatility vs Drawdown
        plt.scatter(
            subset["volatility_30d"],
            subset["max_drawdown_90d"],
            color=colors.get(risk, "blue"),
            label=risk,
            alpha=0.6
        )

    plt.xlabel("30-Day Volatility")
    plt.ylabel("90-Day Max Drawdown")
    plt.title(f"Risk Scatter Plot for {df['ticker'].iloc[0]}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=feature_names)
    
    plt.figure(figsize=(8, 5))
    feature_importance.sort_values().plot(kind="barh")
    plt.title("Feature Importance for Risk Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()