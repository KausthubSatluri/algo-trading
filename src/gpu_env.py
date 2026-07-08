
import torch
import pandas as pd
import numpy as np

class VectorizedStockEnv:
    def __init__(self, df, num_envs=4096, initial_balance=1000.0, transaction_cost=0.001, episode_length=756, device='cuda'):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # 1. Prepare Data Tensors
        # df columns: [SPY, VIX, TNX, GOLD, Returns, RSI, MACD, MACD_Signal, BB_Width, BB_Pos, SMA_50_Ratio, SMA_200_Ratio]
        # We separate 'Price' (for trading) and 'Features' (for input)
        
        self.prices_raw = torch.tensor(df['SPY'].values, dtype=torch.float32, device=self.device)
        
        # All columns as features (normalize data roughly)
        # Z-score normalization for features to help training
        features_np = df.values
        means = np.mean(features_np, axis=0)
        stds = np.std(features_np, axis=0) + 1e-8
        norm_features = (features_np - means) / stds
        
        self.features = torch.tensor(norm_features, dtype=torch.float32, device=self.device)
        self.feature_dim = self.features.shape[1]
        self.max_steps = len(self.prices_raw)
        
        # Episode config
        self.episode_length = episode_length
        self.lookback = 10
        
        # State
        self.env_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.balances = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.shares = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.total_values = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.steps_in_episode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # Obs Dim: 
        # [Balance(1), Shares(1), CurrentPrice(1)] + [Lookback * FeatureDim]
        # Note: Flattening history window
        self.obs_dim = 3 + (self.lookback * self.feature_dim)
        
        self.reset()
        
    def reset(self, env_ids=None):
        max_start = self.max_steps - self.episode_length - 5
        
        # Robustness for when data barely fits one episode
        if max_start <= self.lookback:
            max_start = self.lookback + 1
            
        if env_ids is None:
            # Full reset
            # Randomize start indices
            starts = torch.randint(self.lookback, max_start, (self.num_envs,), device=self.device)
            
            self.env_indices = starts
            self.balances.fill_(self.initial_balance)
            self.shares.fill_(0.0)
            self.steps_in_episode.fill_(0)
            
            prices = self.prices_raw[self.env_indices]
            self.total_values = self.balances + (self.shares * prices)
            
            return self._get_obs()
        else:
            num = len(env_ids)
            starts = torch.randint(self.lookback, max_start, (num,), device=self.device)
            
            self.env_indices[env_ids] = starts
            self.balances[env_ids] = self.initial_balance
            self.shares[env_ids] = 0.0
            self.steps_in_episode[env_ids] = 0
            
            prices = self.prices_raw[self.env_indices[env_ids]]
            self.total_values[env_ids] = self.balances[env_ids] + (self.shares[env_ids] * prices)
            
            return self._get_obs()

    def _get_obs(self):
        # 1. Get History Windows for all Features
        # indices: (N, 1) broadcast to (N, Lookback)
        window_indices = self.env_indices.unsqueeze(1) - torch.arange(self.lookback-1, -1, -1, device=self.device)
        
        # Fetch features: (N, Lookback, FeatureDim)
        history = self.features[window_indices] 
        # Flatten: (N, Lookback * FeatureDim)
        history_flat = history.reshape(self.num_envs, -1)
        
        # 2. Portfolio State
        prices = self.prices_raw[self.env_indices]
        # Normalize balance/shares roughly for the net? 
        # Ideally we logarithmic scale them or relative to initial.
        # Simple scaling:
        b_norm = self.balances / 1000.0
        s_norm = self.shares # Arbitrary scale? Maybe leave raw if net handles it.
        p_norm = prices / 300.0 # Approx avg SPY price.
        
        state_vec = torch.stack([b_norm, s_norm, p_norm], dim=1)
        
        return torch.cat([state_vec, history_flat], dim=1)

    def step(self, actions):
        current_prices = self.prices_raw[self.env_indices]
        prev_values = self.total_values.clone()
        
        # Actions: 0:Hold, 1:Buy10, 2:Sell10, 3:BuyAll, 4:SellAll
        is_buy_10 = (actions == 1)
        is_sell_10 = (actions == 2)
        is_buy_all = (actions == 3)
        is_sell_all = (actions == 4)
        
        cost_mult = (1 + self.transaction_cost)
        rev_mult = (1 - self.transaction_cost)

        # Buy 10%
        if is_buy_10.any():
            spend = self.balances[is_buy_10] * 0.1
            shares_new = spend / (current_prices[is_buy_10] * cost_mult)
            self.balances[is_buy_10] -= spend
            self.shares[is_buy_10] += shares_new

        # Sell 10%
        if is_sell_10.any():
            shares_out = self.shares[is_sell_10] * 0.1
            revenue = shares_out * current_prices[is_sell_10] * rev_mult
            self.shares[is_sell_10] -= shares_out
            self.balances[is_sell_10] += revenue
            
        # Buy All
        if is_buy_all.any():
            spend = self.balances[is_buy_all]
            shares_new = spend / (current_prices[is_buy_all] * cost_mult)
            self.balances[is_buy_all] = 0.0
            self.shares[is_buy_all] += shares_new
            
        # Sell All
        if is_sell_all.any():
            revenue = self.shares[is_sell_all] * current_prices[is_sell_all] * rev_mult
            self.shares[is_sell_all] = 0.0
            self.balances[is_sell_all] += revenue

        # Update
        self.total_values = self.balances + (self.shares * current_prices)
        
        # Reward: Log Return of Portfolio Value
        # r = ln(V_t / V_{t-1})
        # This properly incentivizes compounding growth and is symmetric for gains/losses
        
        # Avoid log(0)
        safe_prev = torch.clamp(prev_values, min=1e-8)
        safe_curr = torch.clamp(self.total_values, min=1e-8)
        
        rewards = torch.log(safe_curr / safe_prev) * 100.0 # Scale up for numerical stability

        
        self.env_indices += 1
        self.steps_in_episode += 1
        
        dones = (self.steps_in_episode >= self.episode_length) | (self.env_indices >= self.max_steps - 1)
        
        info = {}
        if dones.any():
            done_ids = torch.nonzero(dones).flatten()
            # Capture terminal values before reset
            info['terminal_observation'] = self.total_values[done_ids].clone()
            info['terminal_ids'] = done_ids
            info['terminal_prices'] = self.prices_raw[self.env_indices[done_ids]].clone()
            self.reset(done_ids)
            
        return self._get_obs(), rewards, dones, info
