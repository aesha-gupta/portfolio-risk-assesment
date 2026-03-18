# 📈 CoinDCX Crypto Portfolio Risk ML

Welcome to the **CoinDCX Crypto Portfolio Risk Assessor**! This specialized version of the pipeline integrates directly with the live CoinDCX exchange to pull your real cryptocurrency trades and seamlessly evaluate their risk using advanced Machine Learning models.

### Key Upgrades (CoinDCX Crypto Baseline)

- **Live Exchange Sync**: Securely authenticate via your CoinDCX API Key & Secret inside the Streamlit sidebar.
- **Ledger Preprocessing Engine**: Automatically groups messy crypto partial-fills, sums up exact quantities, and mathematically calculates your true *Weighted Average Buy Price* natively.
- **Yahoo Finance Translation**: Auto-maps base cryptocurrencies (like `BTC` and `ETH`) to their global ticker equivalents (like `BTC-USD`) to fetch flawless historical data.
- **Hackathon Sandbox Mode**: Never fail a demo! If you forget your API keys, the system instantly triggers a hyper-realistic Mock Crypto Ledger that successfully passes through the mathematical preprocessing pipeline just like the real API!

### 🌟 6-Part "Ideal Output" Dashboard
Displays incredibly professional outputs tailored for multi-asset crypto portfolios:
1. **Final Risk Summary:** Top-layer human readable Risk Level & Score.
2. **Key Risk Metrics:** Quantitative breakdown (Sharpe, Drawdown, Diversification).
3. **Model Explanation (SHAP):** Live, dynamic factor analysis explicitly explaining exactly why your crypto portfolio is deemed risky.
4. **Model Confidence:** Probability percentages via XGBoost.
5. **Anomaly Detection:** Direct feedback from the `IsolationForest` decision functions hunting for algorithmic crypto drifts.
6. **Actionable Insights:** Live recommendations on how to explicitly hedge current portfolio vulnerabilities (e.g. *Reduce 50% concentration in Pepe*).

### Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
