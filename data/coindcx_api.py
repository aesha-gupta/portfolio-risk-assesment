import time
import hmac
import hashlib
import json
import requests
import pandas as pd
from datetime import datetime

class CoinDCXFetcher:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coindcx.com"

    def _generate_signature(self, payload: dict) -> str:
        secret_bytes = bytes(self.api_secret, encoding='utf-8')
        payload_str = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(secret_bytes, payload_str.encode(), hashlib.sha256).hexdigest()
        return signature

    def fetch_raw_trades(self, limit: int = 500) -> list:
        """
        Hits the CoinDCX trade_history endpoint using HMAC authentication.
        """
        endpoint = "/exchange/v1/orders/trade_history"
        url = self.base_url + endpoint
        
        timestamp = int(round(time.time() * 1000))
        payload = {
            "timestamp": timestamp,
            "limit": limit
        }
        
        signature = self._generate_signature(payload)
        
        headers = {
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, data=json.dumps(payload, separators=(',', ':')), headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"CoinDCX API Error {response.status_code}: {response.text}")

    def extract_base_coin(self, symbol: str) -> str:
        """
        Attempts to extract the base coin (e.g. BTC) from a market symbol (e.g. BTCINR, ETHUSDT)
        """
        symbol = str(symbol).upper()
        if symbol.endswith("INR"): return symbol[:-3]
        if symbol.endswith("USDT"): return symbol[:-4]
        if symbol.endswith("BTC"): return symbol[:-3]
        if symbol.endswith("ETH"): return symbol[:-3]
        if symbol.endswith("BNB"): return symbol[:-3]
        return symbol # fallback
        
    def process_trade_history(self, raw_trades: list) -> pd.DataFrame:
        """
        Converts the messy ledger of trades into a clean, aggregated portfolio mapped to Yahoo Finance.
        Group by coin -> calculate avg buy price -> get total quantity -> earliest date.
        """
        if not raw_trades:
            return pd.DataFrame()
            
        # Flatten into a DataFrame
        df = pd.DataFrame(raw_trades)
        
        # Ensure correct datatypes
        df["price"] = pd.to_numeric(df.get("price", 0))
        df["quantity"] = pd.to_numeric(df.get("quantity", 0))
        
        # Some endpoints return "side" (buy/sell). Let's filter for Buys for average buy price
        # If 'side' does not exist, assume they are all trades that represent taking a position (simplification for missing data)
        if "side" in df.columns:
            df = df[df["side"].str.lower() == "buy"].copy()
            
        if df.empty:
            return pd.DataFrame()
            
        # Parse timestamp
        # CoinDCX timestamps are usually in milliseconds
        if "timestamp" in df.columns:
            df["date_obj"] = pd.to_datetime(df["timestamp"], unit='ms').dt.date
        elif "created_at" in df.columns:
            # Alternate time field depending on exact api version
            df["date_obj"] = pd.to_datetime(df["created_at"]).dt.date
        else:
            # Fallback to current date if missing
            df["date_obj"] = datetime.today().date()
            
        df["base_coin"] = df["symbol"].apply(self.extract_base_coin)
        
        # Calculate Total Cost to deduce accurate Weighted Average Price
        df["total_cost"] = df["price"] * df["quantity"]
        
        portfolio_rows = []
        
        # Group by Base Coin
        for coin, group in df.groupby("base_coin"):
            total_qty = group["quantity"].sum()
            if total_qty <= 0:
                continue
                
            total_cost_spent = group["total_cost"].sum()
            avg_buy_price = total_cost_spent / total_qty
            earliest_date = group["date_obj"].min()
            
            # Map to Yahoo Finance Format:
            # e.g. BTC -> BTC-USD
            yf_ticker = f"{coin}-USD"
            
            portfolio_rows.append({
                "Ticker": yf_ticker,
                "Quantity": total_qty,
                "Average_Buy_Price": avg_buy_price,
                "Date": earliest_date
            })
            
        final_df = pd.DataFrame(portfolio_rows)
        return final_df
