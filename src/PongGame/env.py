from typing import Tuple, Union, List
from dataclasses import dataclass
import numpy as np

from src.PongGame.game import PongGame, Direction

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
        return self.game.get_snapshot().transpose(1,0,2)

    def do_action(self, action: np.ndarray) -> ActionResult:
        self.steps_taken += 1
        
        # Execute action in game
        reward, terminated = self._take_action(action)
        return ActionResult(self.get_state(), reward, terminated, self.game.score)
    
    def get_state(self) -> np.ndarray:
        state = self.game.get_snapshot()
        return np.array(state)

    def _take_action(self, action: np.ndarray) -> Tuple[int, bool]:
        prev_score = self.game.score
        game_over, score = self.game.play_step(action)
        reward = 0
        if game_over:
            reward = -1
        elif score > prev_score:
            reward = +1
        elif score < prev_score:
            reward = -1
        
        return reward, game_over

if __name__ == '__main__':
    game = PongGame()
    env = GameEnvironment(game)
    state = env.get_state()