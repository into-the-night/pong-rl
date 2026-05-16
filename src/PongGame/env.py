"""Gymnasium env wrapping the pure-numpy Pong, plus a vector-env helper."""
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from PongGame.game import PongGame


class PongEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.game = PongGame(seed=seed)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.rng = np.random.default_rng(seed)
        self.game.reset()
        return self.game.get_state_vector(), {}

    def step(self, action):
        obs, reward, terminated = self.game.step(int(action))
        return obs, float(reward), bool(terminated), False, {"score": self.game.score}

    def render(self):
        return self.game.render()


def make_vector_env(num_envs: int, base_seed: int = 0) -> gym.vector.SyncVectorEnv:
    def factory(rank):
        def _thunk():
            return PongEnv(seed=base_seed + rank)
        return _thunk
    return gym.vector.SyncVectorEnv([factory(i) for i in range(num_envs)])
