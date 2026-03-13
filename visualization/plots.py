import matplotlib.pyplot as plt

def plot_risk_distribution(df):

    counts = df["risk_level"].value_counts()

    plt.figure()
    counts.plot(kind="bar")

    plt.title("Portfolio Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Portfolios")

    plt.show()



def plot_anomalies(df):

    counts = df["anomaly"].value_counts()

    plt.figure()
    counts.plot(kind="bar")

    plt.title("Anomaly Detection Results")
    plt.xlabel("Anomaly Label")
    plt.ylabel("Number of Portfolios")

    plt.show()




def plot_risk_scatter(df):

    colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }

    plt.figure()

    for risk in df["risk_level"].unique():
        subset = df[df["risk_level"] == risk]

        plt.scatter(
            subset["volatility"],
            subset["concentration"],
            color=colors[risk],
            label=risk,
            alpha=0.5
        )

    plt.xlabel("Volatility")
    plt.ylabel("Concentration")
    plt.title("Portfolio Risk Distribution")

    plt.legend()
    plt.show()



def plot_dominant_asset_distribution(df):

    counts = df["dominant_asset"].value_counts()

    plt.figure()
    counts.plot(kind="bar")

    plt.title("Dominant Asset Class in Portfolios")
    plt.xlabel("Asset Type")
    plt.ylabel("Number of Portfolios")

    plt.show()



def plot_feature_importance(model, feature_names):

    import matplotlib.pyplot as plt
    import pandas as pd

    importance = model.feature_importances_

    feature_importance = pd.Series(importance, index=feature_names)

    feature_importance.sort_values().plot(kind="barh")

    plt.title("Feature Importance for Risk Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.show()