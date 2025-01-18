from collections import deque
from typing import Union, Tuple, List, Optional
import random
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from src.models.rl_agent.modules import PolicyNetwork, PolicyTrainer
from src.PongGame.env import ActionResult, GameEnvironment

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class ValueForEndGame(Enum):
    last_action = "last_action"
    not_exist = "not_exist"

@dataclass
class PolicyAgentConfig:
    max_memory: int
    batch_size: int
    hidden_state: int
    value_for_end_game: ValueForEndGame
    iterations: int
    min_deaths_to_record: int
    lr: float = 0.01
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    alpha: float = 0.9
    gamma: float = 0.9
    train_every_iteration: int = 10
    save_every_iteration: Optional[int] = None

class PolicyAgent:
    def __init__(
        self,
        env: GameEnvironment,
        config: PolicyAgentConfig,
        model_path: str,
        dataset_path: str,
        last_checkpoint: Optional[str]
    ):
        self.config = config
        self.model_path = model_path
        self.model = PolicyNetwork(len(env.get_state()), self.config.hidden_state, env.actions_length())
        self.trainer = PolicyTrainer(self.model, lr=config.lr, alpha=config.alpha, gamma=config.gamma)
        self.env = env
        self.steps = 0
        self.dataset_path = dataset_path
        self.count_games = 0
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=config.lr, alpha=config.alpha)
        self.recorded_actions = []
        self.epsilon = config.epsilon_start
        self.begin_iteration = 0
        if last_checkpoint:
            parameters = torch.load(last_checkpoint)
            self.model.load_state_dict(parameters["model"])
            self.optimizer.load_state_dict(parameters["optimizer"])
            self.count_games = parameters.get("count_games", 0)
            self.begin_iteration = parameters.get("begin_iteration", 0)

    @property
    def snapshots_path(self):
        return os.path.join(self.dataset_path, "snapshots")

    @property
    def actions_path(self):
        return os.path.join(self.dataset_path, "actions")

    def _get_action(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        state_tensor = torch.tensor(state, dtype=torch.float, requires_grad=True)
        prob = self.model(state_tensor)
        if np.random.uniform() < self.epsilon:
            action = random.randint(0, 1)
        else:
            action = 1 if np.random.uniform() < prob else 0
        return prob, action
    
    def _save_snapshot(self, step: int):
        plt.imsave(os.path.join(self.snapshots_path, f'{step}.jpg'), self.env.get_snapshot())
    
    def _save_actions(self):
        with open(self.actions_path, mode="w") as file:
            file.write("\n".join([str(action) for action in self.recorded_actions]))
    
    def play_step(
        self,
        record: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, ActionResult]:
        old_state = self.env.get_state()
        prob, action = self._get_action(old_state)
        self.steps += 1
        if step is None:
            step = self.steps
        result = self.env.do_action(action)
        # print(result.reward)
        if record:
            self._save_snapshot(step)
            self.recorded_actions.append(action)
            self._save_actions()
        return old_state, action, result, prob

    def train(self, show_plot: bool = False, record: bool = False, clear_old: bool = False):
        self._setup_training(clear_old)
        
        plot_scores = []
        plot_mean_scores = []
        top_result = 0
        total_score = 0
        states, losses, rewards = [],[],[]
        print(f"Begin iteration is {self.begin_iteration}")
        print(f"All iteration is {self.config.iterations}")
        if self.begin_iteration >= self.config.iterations:
            return
        for iteration in range(self.begin_iteration, self.config.iterations):
            old_state, action, result, prob = self.play_step(
                record=record and self.count_games >= self.config.min_deaths_to_record
            )
            reward, new_state, done = result.reward, result.new_state, result.terminated
            # self.memory.push(old_state, action, result.reward, result.new_state, result.terminated)
            # print(f"This is reward {reward} for action {action}")
            states.append(old_state)
            losses.append(torch.tensor(action, dtype=prob.dtype,) - prob)
            rewards.append(reward)

            def do_training(states: List, losses: List, rewards: List):
                states = torch.Tensor(np.vstack(states))
                losses = torch.stack(losses)
                rewards = torch.Tensor(np.vstack(rewards))
                self.trainer.train_step(states, losses, rewards, done)

            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
            if done:
                self.count_games += 1
                score = result.score
                self.env.reset()
                do_training(states, losses, rewards)
                # for name, param in self.model.named_parameters():
                    # if param.requires_grad and param.grad:
                    #     print(name, param.grad.norm())
                if record and self.count_games > self.config.min_deaths_to_record:
                    if self.config.value_for_end_game.value == ValueForEndGame.last_action.value:
                        self.steps += 1
                        self.recorded_actions.append(self.env.actions_length())
                        self._save_snapshot(self.steps)
                    elif self.config.value_for_end_game.value == ValueForEndGame.not_exist.value:
                        pass
                self._save_actions()

                states, losses, rewards = [],[],[]
                if score > top_result:
                    top_result = score
                    self.save_agent(iteration)

                print('Game', self.count_games, 'Score', score, 'Record:', top_result, "Iteration:", iteration)
                if show_plot:
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / self.count_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
            if self.config.save_every_iteration is not None and iteration % self.config.save_every_iteration == 0:
                self.save_agent(iteration)
        self._save_actions()
        self.save_agent(iteration+1)
        print(f"finish iteration is {iteration}")

    def _setup_training(self, clear_old: bool):
        if clear_old:
            self._clear_training_data()
        else:
            self._load_training_data()
        os.makedirs(self.snapshots_path, exist_ok=True)
        if os.path.dirname(self.model_path) != "":
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _clear_training_data(self):
        self.steps = 0
        self.recorded_actions = []
        shutil.rmtree(self.dataset_path)

    def _load_training_data(self):
        try:
            self.steps = len([f for f in os.listdir(self.snapshots_path) if f.endswith('.jpg')])
            with open(self.actions_path) as f:
                self.recorded_actions = [int(line) for line in f]
        except:
            self.steps = 0
            self.recorded_actions = []
        print(self.steps, len(self.recorded_actions))
        assert self.steps == len(self.recorded_actions)

    def save_agent(self, iteration: int):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "count_games": self.count_games,
            "begin_iteration": iteration
        }, self.model_path)