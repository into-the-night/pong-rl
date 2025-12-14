from typing import Tuple, Union, List
from dataclasses import dataclass
import numpy as np
import torch
from PongGame.game import PongGame, Direction

@dataclass
class ActionResult:
    new_state: np.ndarray
    reward: float
    terminated: bool
    score: float

class GameEnvironment():
    def __init__(self, game: PongGame):
        super().__init__()
        self.steps_taken = 0
        self.game = game
        game.reset()

    def actions_length(self) -> int:
        return 2

    def reset(self):
        self.game.reset()
        self.steps_taken = 0

    def get_snapshot(self) -> np.ndarray:
        # Returns (140, 100) grayscale array
        return self.game.get_snapshot()

    def do_action(self, action: np.ndarray) -> ActionResult:
        self.steps_taken += 1
        
        # Execute action in game
        reward, terminated = self._take_action(action)
        return ActionResult(self.get_state(), reward, terminated, self.game.score)
    
    def get_state(self) -> torch.Tensor:
        # Use structured state instead of pixels for faster learning
        # This gives the agent velocity information (ball direction)
        state_vector = self.game.get_state_vector()
        return torch.Tensor(state_vector)

    def _take_action(self, action: np.ndarray) -> Tuple[int, bool]:
        prev_score = self.game.score
        game_over, reward = self.game.play_step(action)
        return reward, game_over

if __name__ == '__main__':
    game = PongGame()
    env = GameEnvironment(game)
    state = env.get_state()