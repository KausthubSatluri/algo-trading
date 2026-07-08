
import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_env import VectorizedStockEnv
from gpu_train import DeepActorCritic
from data_loader import load_data

def evaluate_extended():
    df = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We want 100 simultaneous evals
    NUM_SAMPLES = 100
    env = VectorizedStockEnv(df, num_envs=NUM_SAMPLES, device=device)
    
    agent = DeepActorCritic(env.obs_dim, 5).to(device)
    try:
        agent.load_state_dict(torch.load("models/gpu_ppo_agent_large.pt"))
        print("Loaded Large Model.")
    except:
        print("Model not found. Run gpu_train.py first.")
        return
    
    agent.eval()
    
    obs = env.reset() # This randomizes the 10 start points
    
    # Tracking
    # balance_history: [Steps, Envs]
    balance_history = np.zeros((env.episode_length, NUM_SAMPLES))
    bnh_history = np.zeros((env.episode_length, NUM_SAMPLES))
    
    # Initial setup for B&H comparison
    initial_balances = env.balances.cpu().numpy().copy() # 1000
    start_prices = env.prices_raw[env.env_indices].cpu().numpy().copy()
    bnh_shares = initial_balances / start_prices
    
    print("Running 100 Random 3-Year Simulations...")
    
    for i in range(env.episode_length):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        
        obs, rewards, dones, info = env.step(action)
        
        # Record Current Values
        current_vals = env.total_values.cpu().numpy().copy()
        
        # Patch terminal values if reset happened
        if 'terminal_observation' in info:
            term_ids = info['terminal_ids'].cpu().numpy()
            term_vals = info['terminal_observation'].cpu().numpy()
            current_vals[term_ids] = term_vals

        current_prices = env.prices_raw[env.env_indices].cpu().numpy().copy()
        
        if 'terminal_prices' in info:
             term_prices = info['terminal_prices'].cpu().numpy()
             # We reuse term_ids from above (assuming logic flow is contiguous)
             # But just to be safe let's grab it or rely on ordering if only one block
             # Ideally we use the same indices.
             # term_ids was defined in the 'terminal_observation' block 
             # Let's clean up the block structure in previous step or just repeat lookup
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
    
    print(f"\n--- RESULTS (100 Samples) ---")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Profit over B&H: ${avg_perf:.2f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Equity Curves of Best Run vs B&H
    best_idx = np.argmax(diffs)
    axes[0].plot(balance_history[:, best_idx], label='Bot (Best Run)', color='green')
    axes[0].plot(bnh_history[:, best_idx], label='Buy & Hold', linestyle='--', color='gray')
    axes[0].set_title(f"Best Run (Profit: ${diffs[best_idx]:.2f})")
    axes[0].legend()
    
    # Plot 2: Scatter of Final Results
    axes[1].scatter(final_bnh, final_bot, c='blue', label='Simulations')
    # x=y line
    lims = [
        np.min([axes[1].get_xlim(), axes[1].get_ylim()]),  # min of both axes
        np.max([axes[1].get_xlim(), axes[1].get_ylim()]),  # max of both axes
    ]
    axes[1].plot(lims, lims, 'r-', alpha=0.75, zorder=0, label='InParity')
    axes[1].set_xlabel("Buy & Hold Final $")
    axes[1].set_ylabel("Bot Final $")
    axes[1].set_title("Bot vs B&H Correlation")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_report.png')
    print("Saved evaluation_report.png")

if __name__ == "__main__":
    evaluate_extended()
