# 📈 Single-Stock Machine Learning Risk Assessor

Welcome to the **Single-Stock Risk Assessment** pipeline! This application allows you to input any stock ticker (e.g., AAPL) and its purchase date, and runs it through a massive Machine Learning architecture to evaluate your investment risk.

### Features

- **Live Data Fetching**: Powered by `yfinance`, the app downloads the complete history of your chosen asset to build an extensive training dataset.
- **Advanced Feature Engineering**: Calculates cutting-edge financial metrics natively, including:
  - Rolling 30-Day Volatility
  - 90-Day Maximum Drawdowns
  - 30-Day Momentum
  - Rolling Sharpe Ratio (Risk-Adjusted Reward)
- **Machine Learning Models**: 
  - **Random Forest & XGBoost**: Predicts whether your asset has a High, Medium, or Low future crash risk based on underlying market logic.
  - **Isolation Forest**: Unsupervised anomaly detection that specifically triggers when your stock behaves outside of standard mathematical patterns.
- **Drift Detection**: Analyzes if current market shifts are drifting away from historical norms (utilizing KS-tests).
- **Interactive Data Visualization**: Stunning Plotly integrations to show risk trajectories, anomaly scatter plots, and Feature Importance scoring.

### How to Run

1. Make sure you have the required packages: `pip install -r requirements.txt` (including `streamlit`, `yfinance`, `scikit-learn`, `xgboost`, `plotly`).
2. Run the application locally using Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Input your Stock Ticker, the exact Purchase Date, and your Investment Quantity directly into the sidebar to begin processing.
