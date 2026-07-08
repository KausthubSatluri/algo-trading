
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_loader import load_data
from stock_env import StockTradingEnv

def train_bot():
    # 1. Load Data
    # Fetching SPY data. 
    # NOTE: This relies on daily data upsampled to 15m as per data_loader fallback
    # To strictly follow "15 years" we ask for 2010-2025
    df = load_data("SPY", start="2010-01-01", end="2025-01-01", period="1d")
    
    if df.empty:
        print("No data found. Exiting.")
        return

    # 2. Create Environment
    # We pass the full DF. The Env handles random 3-year sampling in reset().
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    import torch
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Setup Agent
    model = PPO("MlpPolicy", env, verbose=1, device=device)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of training timesteps")
    args = parser.parse_args()

    # 4. Train
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    # 5. Save
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_spy_trader")
    print("Model saved to models/ppo_spy_trader")

if __name__ == "__main__":
    train_bot()
