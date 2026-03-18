import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_risk_over_time(df: pd.DataFrame):
    """Line chart showing portfolio value with risk level colors."""
    fig = px.scatter(
        df, 
        x="Date", 
        y="portfolio_value", 
        color="predicted_risk",
        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
        title=f"Portfolio Value over Time ({df['ticker'].iloc[0]})",
        labels={"portfolio_value": "Portfolio Value ($)", "predicted_risk": "Risk Level"}
    )
    fig.update_traces(mode='lines+markers', marker=dict(size=6))
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    return fig

def plot_risk_scatter_interactive(df: pd.DataFrame):
    """Interactive scatter plot of volatility vs drawdown."""
    fig = px.scatter(
        df, 
        x="volatility_30d", 
        y="max_drawdown_90d", 
        color="predicted_risk",
        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
        hover_data=["Date", "daily_return", "price"],
        title=f"Risk Map (Volatility vs Drawdown)",
        labels={"volatility_30d": "30-Day Volatility", "max_drawdown_90d": "90-Day Max Drawdown"}
    )
    fig.update_layout(template="plotly_dark")
    return fig

def plot_anomalies_interactive(df: pd.DataFrame):
    """Bar chart highlights days where anomalies were detected."""
    anomalies_df = df[df["is_anomaly"] == True]
    
    fig = go.Figure()
    
    # Normal line
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["price"],
        mode="lines", name="Price", line=dict(color="blue", width=2)
    ))
    
    # Anomaly markers
    if not anomalies_df.empty:
        fig.add_trace(go.Scatter(
            x=anomalies_df["Date"], y=anomalies_df["price"],
            mode="markers", name="Anomaly", marker=dict(color="red", size=10, symbol="x")
        ))
        
    fig.update_layout(title="Asset Price with Detected Anomalies", template="plotly_dark")
    return fig

def plot_feature_importance_interactive(model, feature_names):
    """Horizontal bar chart for feature importance."""
    importance = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        df_imp, 
        x="Importance", 
        y="Feature", 
        orientation='h',
        title="Machine Learning Feature Importance",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(template="plotly_dark")
    return fig
