# 📈 Generic Multi-Asset Portfolio Risk ML

Welcome to the **Multi-Asset Portfolio ML Risk Assessor**! This advanced version of the pipeline aggregates any combination of financial assets into a unified master portfolio for global risk tracking, anomaly detection, and deep machine learning predictions.

### Key Upgrades (Generic Baseline)

- **Dynamic Interactive Portfolio Builder**: Uses Streamlit's cutting-edge `data_editor` to allow you to input ANY combination of Tickers (Stocks, Bonds, Index Funds, Crypto), exact Purchase Dates, and Quantities right inside the user interface.
- **Master Aggregation Engine**: Computes your true 'Portfolio Target Value' day-by-day historically, correctly handling overlapping timelines, varied start dates, and changing weights.
- **Centralized Risk Classifiers**:
  - Treats your entire aggregated portfolio as a singular mathematical entity.
  - Predicts overarching **Probability of High Risk** crashes via an XGBoost pipeline.
  - Generates global **Sharpe Ratios**, **Portfolio Volatility**, and **Concentration Penalties**.

### 🌟 6-Part "Ideal Output" Dashboard
Displays incredibly professional outputs for stakeholders:
1. **Final Risk Summary:** Top-layer human readable Risk Level & Score.
2. **Key Risk Metrics:** Quantitative breakdown (Sharpe, Drawdown, Diversification).
3. **Model Explanation (SHAP):** Live, dynamic factor analysis. What exact factors are currently dragging the portfolio into High Risk (e.g. `+0.35 High Volatility`)?
4. **Model Confidence:** Probability percentages via XGBoost.
5. **Anomaly Detection:** Direct feedback from the `IsolationForest` decision functions.
6. **Actionable Insights:** Live recommendations on how to explicitly hedge current portfolio vulnerabilities (e.g. *Reduce 60% concentration in TSLA*).

### Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
