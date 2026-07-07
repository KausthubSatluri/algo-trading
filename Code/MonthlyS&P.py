import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1) Get daily total-return prices (adjusted for splits & dividends)
spx = yf.download("^GSPC", period="20y", interval="1d", auto_adjust=True)["Close"]

# 2) Pick a contribution schedule: month-end close
px_m = spx.resample("M").last()              # last trading day of each month
contrib = pd.Series(100.0, index=px_m.index) # $100 each month

# 3) Simulate DCA: buy shares = dollars / price; keep running share count
shares_bought = contrib / px_m
cum_shares = shares_bought.cumsum()

# 4) Portfolio value over time (with dividends via adjusted prices)
portfolio_value = cum_shares * px_m
total_contributed = contrib.cumsum()

# 5) Plot
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value (DCA $100/mo)")
plt.plot(total_contributed.index, total_contributed, label="Total Contributions", linestyle="--")
plt.title("DCA: $100/month into S&P 500 (20 Years, Total Return via Adjusted Prices)")
plt.xlabel("Year")
plt.ylabel("USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# (Optional) quick summary
final_value = portfolio_value.iloc[-1]
invested = total_contributed.iloc[-1]
gain_pct = (final_value / invested - 1) * 100
print(f"Invested: ${invested:,.0f}  |  Final: ${final_value:,.0f}  |  Gain: {gain_pct:.1f}%")