import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1) Download AAPL data from 2023-01-01 to today
df = yf.download("AAPL", start="2023-01-01", auto_adjust=True)

# 2) Calculate daily return (Close - Open) / Open
df["DailyReturn"] = (df["Close"] - df["Open"]) / df["Open"]

# 3) Calculate total return over the period
total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1

# 4) Filter out NaNs
returns = df["DailyReturn"].dropna()

# 5) Summary statistics
summary_stats = {
    "Mean": returns.mean(),
    "Median": returns.median(),
    "Std Dev": returns.std(),
    "Min": returns.min(),
    "Max": returns.max(),
    "Count": returns.count(),
    "Total Return": total_return
}
summary_df = pd.DataFrame(summary_stats, index=["Value"])
print(summary_df)

# 6) Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, edgecolor='black', alpha=0.7)
plt.title("AAPL Daily Returns Histogram (2023–2025)")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()