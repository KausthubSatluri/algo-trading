# AlgoTrading

Quantitative finance projects in Python, from market-data analysis to a reinforcement-learning trading agent trained with PPO.

## RL Trading Agent

A PPO (proximal policy optimization) agent trained to trade SPY, with a GPU-vectorized environment for large-scale parallel rollouts.

- `src/gpu_env.py` — `VectorizedStockEnv`: batched trading environment running thousands of episodes in parallel on GPU (2048 envs per rollout).
- `src/gpu_train.py` — PPO training loop with GAE, a deep actor-critic network, and full-episode rollouts.
- `src/train_autonomous.py` — autonomous training driver that alternates training and evaluation, logging win rate and profit vs. buy-and-hold each iteration (`optimization_log.csv`, `optimization_progress.png`).
- `src/gpu_evaluate.py` / `src/gpu_evaluate_unseen.py` — in-sample and out-of-sample evaluation (`evaluation_report.png`, `evaluation_report_oos.png`).
- `src/data_loader.py` — market data loading and preprocessing.
- `legacy/` — the original single-environment (CPU) implementation the GPU version was scaled up from.
- `models/` — trained agent checkpoints.

## Market Analysis

- `Code/1000S&P.py` — downloads equity price history, computes daily and total returns, and produces summary statistics and return distributions.
- `Code/MonthlyS&P.py` — simulates a dollar-cost-averaging strategy on the S&P 500 over 20 years of adjusted prices (monthly contributions, dividend-adjusted).
- `Code/prelim.ipynb` — exploratory analysis notebook.
- `Plots/` — generated figures.

## Stack

Python, PyTorch, pandas, yfinance, matplotlib.
