
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from gpu_env import VectorizedStockEnv
from data_loader import load_data
from gpu_train import DeepActorCritic 

# --- Autonomous Config ---
ITERATION_SIZE = 1 # How many "passes" through the data before evaluating
NUM_EVAL_EPISODES = 50 # Speed up eval
EVAL_INTERVAL = 1 # Evaluate every N iterations

# --- Hyperparams (Tuned) ---
LR = 3e-4 # Slightly higher start
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.02 # Boost exploration slightly
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Reduced Batch Size logic
NUM_ENVS = 128     # Much smaller than 2048
NUM_STEPS = 756    # 3 years
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 4096 
NUM_EPOCHS = 4

TOTAL_ITERATIONS = 20 # How many loops of (Train -> Eval) to run

def evaluate_internal(agent, device):
    """
    Internal fast evaluation on Unseen data (2000-2010).
    Returns mean_reward, win_rate, avg_profit_vs_bnh
    """
    # Load unseen data ONCE if possible, but here we load inside for safety/simplicity
    # To optimize, we should cache this outside the loop.
    # For now, let's assume load_data is fast enough (cached CSV).
    df = load_data(start="2000-01-01", end="2010-01-01") 
    
    env = VectorizedStockEnv(df, num_envs=NUM_EVAL_EPISODES, device=device, episode_length=756)
    
    obs = env.reset()
    
    # Tracking
    final_bot = torch.zeros(NUM_EVAL_EPISODES, device=device)
    final_bnh = torch.zeros(NUM_EVAL_EPISODES, device=device)
    
    # Simple evaluation loop
    # We only care about final results for metrics
    initial_balances = env.balances.clone()
    start_prices = env.prices_raw[env.env_indices].clone()
    bnh_shares = initial_balances / start_prices

    agent.eval()
    
    # Run full episodes
    # Since vectorized, we just run max steps. 
    # Dones might happen early if data ends, but VectorizedStockEnv auto-resets.
    # We need to handle this carefully: we want ONE episode per env.
    
    # Actually, the simplest way for metric tracking in eval is to just run X steps.
    # If an env resets, we might lose the "final" value of the intended episode.
    # gpu_env.py resets automatically.
    
    # Let's run for EXACTLY 756 steps and take the values at step 755.
    
    for _ in range(756):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        obs, _, _, _ = env.step(action)
        
    final_bot = env.total_values
    current_prices = env.prices_raw[env.env_indices]
    final_bnh = bnh_shares * current_prices
    
    # Metrics
    # Move to CPU
    bot_vals = final_bot.cpu().numpy()
    bnh_vals = final_bnh.cpu().numpy()
    
    diffs = bot_vals - bnh_vals
    wins = diffs > 0
    win_rate = np.mean(wins) * 100
    avg_profit = np.mean(diffs)
    
    agent.train() # Switch back to train
    return win_rate, avg_profit

def train_autonomous():
    print("--- Starting Autonomous Optimization ---")
    
    # 1. Setup
    df_train = load_data(start="2010-01-01", end="2025-12-31")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    env = VectorizedStockEnv(df_train, num_envs=NUM_ENVS, device=device)
    agent = DeepActorCritic(env.obs_dim, 5).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    # Load previous if exists?
    # agent.load_state_dict(torch.load("models/gpu_ppo_agent_large.pt"))
    # Let's start FRESH for optimization experiments usually.
    
    # Logs
    history = [] # List of dicts
    
    # Buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, env.obs_dim), device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    global_step = 0
    
    pbar = tqdm(range(TOTAL_ITERATIONS), desc="Auto Loop")
    
    for iteration in pbar:
        # --- TRAINING ---
        # Rollout
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, next_done, info = env.step(action)
            rewards[step] = reward
            
        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in range(NUM_STEPS - 1, -1, -1):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values
            
        # PPO Update
        b_obs = obs.reshape((-1, env.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        b_inds = np.arange(BATCH_SIZE)
        
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # --- EVALUATION ---
        if iteration % EVAL_INTERVAL == 0:
            win_rate, avg_profit = evaluate_internal(agent, device)
            
            log_entry = {
                "Iteration": iteration,
                "WinRate": win_rate,
                "AvgProfitVsBnH": avg_profit,
                "TrainReward": rewards.mean().item()
            }
            history.append(log_entry)
            
            pbar.set_postfix({"Win%": f"{win_rate:.1f}", "Pft": f"${avg_profit:.0f}"})
            
            # Save Checkpoint
            torch.save(agent.state_dict(), f"models/auto_opt_agent.pt")
            
            # Save CSV
            pd.DataFrame(history).to_csv("optimization_log.csv", index=False)
            
    print("Optimization Complete.")
    
    # Final Plot
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['Iteration'], history_df['WinRate'], label='Win Rate %')
    plt.axhline(50, color='r', linestyle='--', label='50% Target')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate %')
    plt.title('Optimization Progress on Unseen Data')
    plt.legend()
    plt.savefig('optimization_progress.png')

if __name__ == "__main__":
    train_autonomous()
