import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class FuturesEnv(gym.Env):
    """
    Env discret 3 actions: 0=short, 1=flat, 2=long
    Reward = PnL net (avec frais) par step.
    Position persistante; frais appliqués sur changement de sens/entrée.
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, fee=0.00005, window=32):
        super().__init__()
        assert {"close","high","low"}.issubset(df.columns)
        self.df = df.reset_index(drop=True)
        self.fee = fee
        self.window = window

        # obs: window ret log + vol (rolling std) + normalisation prix locale
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window+2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.i = max(self.window, 64)
        self.position = 0   # -1,0,1
        self.entry = 0.0
        self.equity = 1.0

    def _obs(self):
        closes = self.df["close"].iloc[self.i-self.window:self.i].values
        rets = np.diff(np.log(closes), prepend=closes[0])
        vol = np.std(rets)
        norm = (closes[-1] - closes.mean()) / (closes.std() + 1e-9)
        x = np.concatenate([rets.astype(np.float32), [vol, norm]]).astype(np.float32)
        return x

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action:int):
        prev_close = float(self.df["close"].iloc[self.i-1])
        price = float(self.df["close"].iloc[self.i])

        reward = 0.0
        # frais si on change de position
        def trade_cost():
            return self.fee

        # exécution
        if action == 2 and self.position != 1:           # entrer long
            if self.position == -1:                      # fermer short -> coût
                reward += (self.entry/price - 1.0) - trade_cost()
            self.position = 1
            self.entry = price
            reward -= trade_cost()
        elif action == 0 and self.position != -1:        # entrer short
            if self.position == 1:
                reward += (price/self.entry - 1.0) - trade_cost()
            self.position = -1
            self.entry = price
            reward -= trade_cost()
        # action == 1 => flat
        elif action == 1 and self.position != 0:
            # fermer la position
            if self.position == 1:
                reward += (price/self.entry - 1.0) - trade_cost()
            elif self.position == -1:
                reward += (self.entry/price - 1.0) - trade_cost()
            self.position = 0
            self.entry = 0.0

        # PnL flottant (tenir une position)
        if self.position == 1:
            reward += (price/prev_close - 1.0)
        elif self.position == -1:
            reward += (prev_close/price - 1.0)

        self.equity *= (1.0 + reward)
        self.i += 1
        terminated = self.i >= len(self.df)-1
        truncated = False
        return self._obs(), float(reward), terminated, truncated, {}
