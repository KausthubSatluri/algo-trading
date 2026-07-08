
import yfinance as yf
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, lower

def load_data(start="2010-01-01", end="2025-12-31"):
    """
    Loads SPY + Macro indicators. Computes Technicals.
    Returns a normalized, aligned DataFrame.
    """
    file_path = os.path.join(DATA_DIR, f"training_data_expanded_{start}_{end}.csv")
    
    if os.path.exists(file_path):
        print(f"Loading cached expanded data from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    print("Downloading Data (SPY, ^VIX, ^TNX, GC=F)...")
    tickers = ["SPY", "^VIX", "^TNX", "GC=F"] # SP500, Volatility, 10Y Yield, Gold
    
    raw = yf.download(tickers, start=start, end=end, interval="1d")
    
    # Flatten MultiIndex if needed
    if isinstance(raw.columns, pd.MultiIndex):
        # We want Close prices mostly.
        # Structure: PriceType -> Ticker
        # or Ticker -> PriceType
        # yfinance > 0.2: (Price, Ticker)
        try:
            closes = raw['Close']
        except:
            # Fallback
            closes = raw.xs('Close', axis=1, level=0)
    else:
        closes = raw

    # Forward fill missing macro data
    closes = closes.ffill().dropna()
    
    df = pd.DataFrame(index=closes.index)
    df['SPY'] = closes['SPY']
    df['VIX'] = closes['^VIX']
    df['TNX'] = closes['^TNX']
    df['GOLD'] = closes['GC=F']
    
    print("Computing Technical Indicators...")
    # 1. Returns
    df['Returns'] = df['SPY'].pct_change()
    
    # 2. RSI
    df['RSI'] = calculate_rsi(df['SPY'])
    
    # 3. MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['SPY'])
    
    # 4. Bollinger
    upper, lower = calculate_bollinger(df['SPY'])
    df['BB_Width'] = (upper - lower) / df['SPY']
    df['BB_Pos'] = (df['SPY'] - lower) / (upper - lower)
    
    # 5. SMA Ratios
    df['SMA_50_Ratio'] = df['SPY'] / df['SPY'].rolling(window=50).mean()
    df['SMA_200_Ratio'] = df['SPY'] / df['SPY'].rolling(window=200).mean()
    
    # Drop NaNs from warm-up
    df = df.dropna()
    
    # Normalize robustness logic should be in Env usually, but we can pre-scale basics here
    # or just keep raw values. Neural nets need scaling. 
    # Let's keep RAW values here so Env can simulate actual prices ($1000 balance etc),
    # but Env observation will normalize.
    
    df.to_csv(file_path)
    print(f"Data ready: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.tail())
