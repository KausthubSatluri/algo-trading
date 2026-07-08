import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    A custom stock trading environment.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=1000, lookback_window=10, transaction_cost_pct=0.001):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost_pct = transaction_cost_pct
        
        # Clean dataframe
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
            
        if 'Close' not in self.df.columns and len(self.df.columns) > 0:
            self.prices = self.df.iloc[:, 0].values
        else:
            self.prices = self.df['Close'].values

        self.n_steps = len(self.prices)
        
        # Action Space:
        # 0: Hold
        # 1: Buy (10% of Cash)
        # 2: Sell (10% of Shares)
        # 3: Buy (100% - All in)
        # 4: Sell (100% - Liquidate)
        self.action_space = spaces.Discrete(5)
        
        # Observation Space:
        # [Balance, Shares, Current Price, ...Last N Prices...]
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(3 + lookback_window,), 
            dtype=np.float32
        )

        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_asset_value = initial_balance
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random 3 year sample or full
        # 3 years * 252 days = 756 steps
        episode_length = 756
        
        if len(self.prices) > episode_length + self.lookback_window + 10:
            max_start = len(self.prices) - episode_length
            self.start_step = np.random.randint(self.lookback_window, max_start)
            self.end_step = self.start_step + episode_length
            self.current_step = self.start_step
        else:
            self.start_step = self.lookback_window
            self.end_step = len(self.prices) - 1
            self.current_step = self.start_step
            
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset_value = self.initial_balance
        self.history = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        price_window = self.prices[self.current_step - self.lookback_window : self.current_step]
        obs = np.array([
            self.balance,
            self.shares_held,
            self.prices[self.current_step],
            *price_window
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.prices[self.current_step]
        cost = 0
        
        # Execute Action
        if action == 1: # Buy 10%
            spend_target = self.balance * 0.1
            if spend_target > current_price:
                # Actual spend is target minus cost? or we pay cost on top?
                # Usually: Cost is taken from balance on top or reduced from purchasing power.
                # Let's say we spend `spend_target` TOTAL. Cost is included.
                # shares * price * (1 + cost_pct) = spend_target
                # shares = spend_target / (price * (1+cost_pct))
                shares = spend_target / (current_price * (1 + self.transaction_cost_pct))
                cost = shares * current_price * self.transaction_cost_pct
                self.balance -= spend_target
                self.shares_held += shares
                
        elif action == 2: # Sell 10%
            shares_to_sell = self.shares_held * 0.1
            if shares_to_sell > 0:
                revenue_gross = shares_to_sell * current_price
                cost = revenue_gross * self.transaction_cost_pct
                revenue_net = revenue_gross - cost
                self.shares_held -= shares_to_sell
                self.balance += revenue_net
                
        elif action == 3: # Buy All
            spend_target = self.balance
            if spend_target > current_price:
                shares = spend_target / (current_price * (1 + self.transaction_cost_pct))
                cost = shares * current_price * self.transaction_cost_pct
                self.balance -= spend_target
                self.shares_held += shares
                
        elif action == 4: # Sell All
            revenue_gross = self.shares_held * current_price
            cost = revenue_gross * self.transaction_cost_pct
            revenue_net = revenue_gross - cost
            self.shares_held = 0
            self.balance += revenue_net

        # Update Value
        prev_asset_value = self.total_asset_value
        self.total_asset_value = self.balance + (self.shares_held * current_price)
        
        # Reward
        reward = (self.total_asset_value - prev_asset_value)
        # Cost is implicitly punished because it reduces asset value/balance
        # Normalizing reward
        reward = reward / self.initial_balance * 1000 
        
        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = False
        
        info = {
            'total_value': self.total_asset_value,
            'price': current_price,
            'cost_paid': cost
        }
        
        return self._next_observation(), reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}, Total: {self.total_asset_value:.2f}")
