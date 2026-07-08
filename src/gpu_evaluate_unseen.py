
import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_env import VectorizedStockEnv
from gpu_train import DeepActorCritic
from data_loader import load_data

def evaluate_unseen_random_sampling():
    # 1. Load UNSEEN Data (2000-2010)
    # Covers Dot Com Bubble (2000-2002) and Housing Crisis (2007-2009)
    print("Loading Unseen Data (2000-2010)...")
    df = load_data(start="2000-01-01", end="2010-01-01")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_SAMPLES = 100
    EPISODE_LENGTH = 756 # 3 Years
    
    print(f"Data Length: {len(df)}")
    print(f"Running {NUM_SAMPLES} Random 3-Year Simulations (2000-2010)...")
    
    # Initialize Env with 100 environments
    env = VectorizedStockEnv(df, num_envs=NUM_SAMPLES, device=device, episode_length=EPISODE_LENGTH)
    
    agent = DeepActorCritic(env.obs_dim, 5).to(device)
    try:
        agent.load_state_dict(torch.load("models/gpu_ppo_agent_large.pt", map_location=device))
        print("Loaded Large Model.")
    except:
        print("Model not found. Run gpu_train.py first.")
        return
    
    agent.eval()
    
    # Reset (Randomizes start points automatically in this 10-year window)
    obs = env.reset()
    
    # Tracking
    balance_history = np.zeros((EPISODE_LENGTH, NUM_SAMPLES))
    bnh_history = np.zeros((EPISODE_LENGTH, NUM_SAMPLES))
    
    initial_balances = env.balances.cpu().numpy().copy()
    start_prices = env.prices_raw[env.env_indices].cpu().numpy().copy()
    bnh_shares = initial_balances / start_prices
    
    for i in range(EPISODE_LENGTH):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        
        obs, rewards, dones, info = env.step(action)
        
        current_vals = env.total_values.cpu().numpy().copy()
        
        # Patch terminal values if done (shouldn't happen if data > 3 years, but good safety)
        if 'terminal_observation' in info:
            term_ids = info['terminal_ids'].cpu().numpy()
            term_vals = info['terminal_observation'].cpu().numpy()
            current_vals[term_ids] = term_vals

        current_prices = env.prices_raw[env.env_indices].cpu().numpy().copy()
        if 'terminal_prices' in info:
             term_prices = info['terminal_prices'].cpu().numpy()
             term_ids = info['terminal_ids'].cpu().numpy()
             current_prices[term_ids] = term_prices
        
        balance_history[i] = current_vals
        bnh_history[i] = bnh_shares * current_prices
        
    # Analysis
    final_bot = balance_history[-1]
    final_bnh = bnh_history[-1]
    
    wins = final_bot > final_bnh
    win_rate = np.mean(wins) * 100
    
    diffs = final_bot - final_bnh
    avg_perf = np.mean(diffs)
    
    print(f"\n--- OOS RESULTS (2000-2010) ---")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Profit over B&H: ${avg_perf:.2f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Equity Curves (Best Run)
    best_idx = np.argmax(diffs)
    axes[0].plot(balance_history[:, best_idx], label='Bot (Best)', color='green')
    axes[0].plot(bnh_history[:, best_idx], label='Buy & Hold', linestyle='--', color='gray')
    axes[0].set_title(f"Best Run (Profit: ${diffs[best_idx]:.2f})")
    axes[0].legend()
    
    # Scatter
    axes[1].scatter(final_bnh, final_bot, c='red', label='OOS Samples')
    lims = [
        np.min([axes[1].get_xlim(), axes[1].get_ylim()]),
        np.max([axes[1].get_xlim(), axes[1].get_ylim()]),
    ]
    axes[1].plot(lims, lims, 'r-', alpha=0.75, zorder=0, label='InParity')
    axes[1].set_xlabel("Buy & Hold Final $")
    axes[1].set_ylabel("Bot Final $")
    axes[1].set_title("OOS Bot vs B&H (2000-2010)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_report_oos.png')
    print("Saved evaluation_report_oos.png")

if __name__ == "__main__":
    evaluate_unseen_random_sampling()
