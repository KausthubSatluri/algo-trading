
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
from tqdm import tqdm
from data_loader import load_data
from gpu_env import VectorizedStockEnv
import os

# --- Scaled up Hyperparams ---
LR = 1e-4 # Slower learning rate for larger network
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# User definition: Iteration = Pass through 3 years (756 steps)
# We will set batch size to roughly match this or multiples.
NUM_STEPS = 756     # FULL EPISODE per rollout
NUM_ENVS = 2048     # Parallel episodes
# total_steps_per_update = 2048 * 756 = ~1.5M steps.
# That's a huge batch. PPO usually updates more frequently.
# But for "1 pass" logic, we can do this.
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 8192 # Larger chunks for GPU efficiency
NUM_EPOCHS = 5 # Reduced from 100 to prevent overfitting

TOTAL_EPISODES = 100_000 # The number of 3-year episodes to train over
TOTAL_TIMESTEPS = TOTAL_EPISODES * NUM_STEPS # Calculated automatically

class DeepActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Simplified Network for Generalization: [256, 128]
        # Switched to Tanh which is often more stable for PPO
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def train_gpu():
    print("Loading Expanded Data (Macro + Technicals)...")
    df = load_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Initializing {NUM_ENVS} parallel environments...")
    env = VectorizedStockEnv(df, num_envs=NUM_ENVS, device=device)
    
    print(f"Observation Dim: {env.obs_dim} (History x Features)")
    
    agent = DeepActorCritic(env.obs_dim, 5).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    # Pre-allocate buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, env.obs_dim), device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    
    global_step = 0
    start_time = time.time()
    
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    
    print(f"Starting Large Scale Training... ({TOTAL_EPISODES} Episodes / {num_updates} Iterations)")
    
    # Outer progress bar for Updates (Episodes)
    pbar_updates = tqdm(range(1, num_updates + 1), desc="Training Progress", unit="iter")
    
    for update in pbar_updates:
        # 1. Rollout (Full 3-year episode per env)
        for step in range(0, NUM_STEPS):
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
            
        # 2. Advantage
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
            
        # 3. Optimize
        b_obs = obs.reshape((-1, env.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        b_inds = np.arange(BATCH_SIZE)
        
        # Inner loop (Epochs) is fast, maybe we don't need a bar, but let's add a small description to outer bar
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
                # Normalize advantage batch
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
        
        # Log
        sps = int(global_step / (time.time() - start_time))
        avg_return = rewards.sum(dim=0).mean().item()
        
        # Update progress bar description
        pbar_updates.set_postfix({"SPS": sps, "Reward": f"${avg_return:.2f}"})
        
        # print(f"Iteration {update}/{num_updates} | SPS: {sps} | Avg Episode Reward: {avg_return:.2f}")
        
        if update % 5 == 0:
            torch.save(agent.state_dict(), "models/gpu_ppo_agent_large.pt")

    # Save final
    torch.save(agent.state_dict(), "models/gpu_ppo_agent_large.pt")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train_gpu()
