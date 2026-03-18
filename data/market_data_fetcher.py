import yfinance as yf
import pandas as pd
from datetime import datetime

class MarketDataFetcher:
    """Class to fetch historical market data via yfinance."""
    
    @staticmethod
    def get_full_history(ticker: str) -> pd.DataFrame:
        """
        Fetches the full available history for a ticker to serve as training data.
        Args:
            ticker: The stock symbol (e.g. 'AAPL')
        Returns:
            A pandas DataFrame of historical daily prices.
        """
        print(f"Downloading full history for {ticker}...")
        stock = yf.Ticker(ticker)
        # Use a long history (e.g. max)
        hist = stock.history(period="max")
        # yfinance returns timezone-aware indexes. Let's make it tz-naive for simplicity later if needed
        hist.index = hist.index.tz_localize(None)
        
        # We only really need the 'Close' or 'Adj Close'
        # Yfinance 'history' method usually adjusts the 'Close' column automatically for splits and dividends
        # if auto_adjust=True (which is default). So 'Close' is functionally 'Adj Close'.
        df = hist[['Close', 'Volume']].copy()
        df.rename(columns={'Close': 'price', 'Volume': 'volume'}, inplace=True)
        # Reset index to make 'Date' a column
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        
        return df

    @staticmethod
    def get_evaluation_window(ticker: str, start_date_str: str) -> pd.DataFrame:
        """
        Fetches the specific user's holding period for evaluation.
        Args:
            ticker: The stock symbol.
            start_date_str: The purchase date (e.g. '2023-01-15 10:30:00')
        """
        print(f"Downloading evaluation window for {ticker} since {start_date_str}...")
        
        # Parse the start date. We only need the YYYY-MM-DD for yfinance daily data
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date)
        hist.index = hist.index.tz_localize(None)
        
        df = hist[['Close', 'Volume']].copy()
        df.rename(columns={'Close': 'price', 'Volume': 'volume'}, inplace=True)
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        
        return df

if __name__ == "__main__":
    # Test
    fetcher = MarketDataFetcher()
    df_train = fetcher.get_full_history("AAPL")
    print("\nTraining Data Head:")
    print(df_train.head())
    
    df_eval = fetcher.get_evaluation_window("AAPL", "2023-01-15 10:30:00")
    print("\nEvaluation Data Head:")
    print(df_eval.head())
