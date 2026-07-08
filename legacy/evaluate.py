import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from data_loader import load_data
from stock_env import StockTradingEnv
import numpy as np

def evaluate_bot():
    df = load_data("SPY", start="2010-01-01", end="2025-01-01", period="1d")
    
    # Create a fresh env for eval
    # Note: reset() will pick a random 3 year chunk. 
    # For consistent comparison, we might want to fix the seed or force a slice, 
    # but the prompt implies general "better than holding" testing.
    env = StockTradingEnv(df)
    
    # Load model
    try:
        model = PPO.load("models/ppo_spy_trader")
    except:
        print("Model not found. Run train.py first.")
        return

    obs, info = env.reset()
    done = False
    
    # Trackers
    portfolio_values = []
    buy_hold_values = []
    
    initial_price = env.prices[env.start_step]
    initial_balance = env.initial_balance
    
    # Buy and Hold: Buy max shares at start
    bnh_shares = initial_balance / initial_price
    
    print("Running evaluation episode...")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        current_val = info['total_value']
        current_price = info['price']
        
        portfolio_values.append(current_val)
        buy_hold_values.append(bnh_shares * current_price)

    # Results
    final_rl = portfolio_values[-1]
    final_bnh = buy_hold_values[-1]
    
    print(f"Evaluation Complete.")
    print(f"RL Bot Final Balance: ${final_rl:.2f}")
    print(f"Buy & Hold Final Balance: ${final_bnh:.2f}")
    
    if final_rl > final_bnh:
        print("SUCCESS: RL Bot outperformed Buy & Hold!")
    else:
        print("FAIL: RL Bot underperformed.")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='RL Bot')
    plt.plot(buy_hold_values, label='Buy & Hold', linestyle='--')
    plt.title('RL Bot vs Buy & Hold (3 Year Sample)')
    plt.legend()
    plt.savefig('performance.png')
    print("Performance plot saved to performance.png")

if __name__ == "__main__":
    evaluate_bot()
