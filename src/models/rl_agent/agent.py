from typing import Union, Tuple, List, Optional
import random
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.distributions import Categorical

from models.rl_agent.modules import PolicyNetwork, ValueNetwork, PPOTrainer
from PongGame.env import ActionResult, GameEnvironment


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

def plot_losses(policy_losses, value_losses, entropies, total_losses, mean_rewards):
    """Plot all loss metrics and rewards after training"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Policy Loss
    axes[0, 0].plot(policy_losses, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Policy Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value Loss
    axes[0, 1].plot(value_losses, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Value Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy
    axes[0, 2].plot(entropies, 'g-', linewidth=1.5)
    axes[0, 2].set_title('Entropy', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Training Step')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Total Loss
    axes[1, 0].plot(total_losses, 'm-', linewidth=1.5)
    axes[1, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean Rewards per Training Step
    axes[1, 1].plot(mean_rewards, 'orange', linewidth=1.5)
    axes[1, 1].set_title('Mean Reward per Training Step', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Mean Reward')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Cumulative Reward
    cumulative_rewards = np.cumsum(mean_rewards)
    axes[1, 2].plot(cumulative_rewards, 'cyan', linewidth=1.5)
    axes[1, 2].set_title('Cumulative Reward', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Training Step')
    axes[1, 2].set_ylabel('Cumulative Reward')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    print(f"\nLoss and reward plots saved to 'training_losses.png'")
    plt.show()

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
    lr: float = 3e-4
    epsilon_start: float = 0.6
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    train_every_iteration: int = 10
    save_every_iteration: Optional[int] = None

class RolloutBuffer:
    """Buffer for storing trajectories for PPO training"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self):
        """Convert buffer contents to tensors"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.values)
        )
    
    def __len__(self):
        return len(self.states)

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
        
        # PPO uses separate actor and critic networks
        self.actor = PolicyNetwork(len(env.get_state()), self.config.hidden_state, env.actions_length())
        self.critic = ValueNetwork(len(env.get_state()), self.config.hidden_state)
        
        # PPO Trainer
        self.trainer = PPOTrainer(
            actor=self.actor,
            critic=self.critic,
            lr=config.lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            ppo_epochs=config.ppo_epochs
        )
        
        self.env = env
        self.steps = 0
        self.dataset_path = dataset_path
        self.count_games = 0
        self.recorded_actions = []
        self.epsilon = config.epsilon_start
        self.begin_iteration = 0
        self.rollout_buffer = RolloutBuffer()
        
        # Loss tracking
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.total_losses = []
        self.mean_rewards = []  # Track mean reward per training step
        
        if last_checkpoint:
            parameters = torch.load(last_checkpoint)
            self.actor.load_state_dict(parameters["actor"])
            self.critic.load_state_dict(parameters["critic"])
            self.trainer.optimizer.load_state_dict(parameters["optimizer"])
            self.count_games = parameters.get("count_games", 0)
            self.begin_iteration = parameters.get("begin_iteration", 0)

    @property
    def snapshots_path(self):
        return os.path.join(self.dataset_path, "snapshots")

    @property
    def actions_path(self):
        return os.path.join(self.dataset_path, "actions")

    def _get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Get action using current policy
        
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            value: State value estimate
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action probabilities from actor
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Sample from policy (NO epsilon-greedy for PPO!)
        # PPO uses entropy bonus for exploration instead
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def _save_snapshot(self, step: int):
        plt.imsave(os.path.join(self.snapshots_path, f'{step}.jpg'), self.env.get_snapshot())
    
    def _save_actions(self):
        with open(self.actions_path, mode="w") as file:
            file.write("\n".join([str(action) for action in self.recorded_actions]))
    
    def play_step(
        self,
        record: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, int, ActionResult, float, float]:
        """Play one step and return state, action, result, log_prob, value"""
        old_state = self.env.get_state()
        action, log_prob, value = self._get_action(old_state)
        
        self.steps += 1
        if step is None:
            step = self.steps
        result = self.env.do_action(action)

        if record:
            self._save_snapshot(step)
            self.recorded_actions.append(action)
            self._save_actions()
        
        return old_state, action, result, log_prob, value

    def train(self, show_plot: bool = False, record: bool = False, clear_old: bool = False, max_games: Optional[int] = None):
        self._setup_training(clear_old)
        
        plot_scores = []
        plot_mean_scores = []
        top_result = 0
        total_score = 0
        
        print(f"Begin iteration is {self.begin_iteration}")
        print(f"All iteration is {self.config.iterations}")
        if max_games:
            print(f"Training will stop after {max_games} games")
        if self.begin_iteration >= self.config.iterations:
            return
        
        for iteration in range(self.begin_iteration, self.config.iterations):
            # Check if we've reached max games
            if max_games and self.count_games >= max_games:
                print(f"\nReached maximum of {max_games} games. Stopping training...")
                break
            old_state, action, result, log_prob, value = self.play_step(
                record=record and self.count_games >= self.config.min_deaths_to_record
            )
            reward, new_state, done = result.reward, result.new_state, result.terminated
            
            # Add to rollout buffer
            self.rollout_buffer.add(old_state, action, reward, done, log_prob, value)

            # Train when buffer is full
            if len(self.rollout_buffer) >= self.config.batch_size and iteration % self.config.train_every_iteration == 0:
                # Get next state value for GAE computation (bootstrap value)
                if not done:
                    next_state_tensor = torch.FloatTensor(new_state)
                    with torch.no_grad():
                        next_value = self.critic(next_state_tensor).item()
                else:
                    next_value = 0.0
                
                # Get all data from buffer
                states, actions, log_probs, rewards, dones, values = self.rollout_buffer.get()
                
                # PPO training step
                loss_info = self.trainer.train_step(
                    states=states,
                    actions=actions,
                    old_log_probs=log_probs,
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    next_value=next_value
                )
                
                # Track losses and rewards
                if loss_info:
                    self.policy_losses.append(loss_info['policy_loss'])
                    self.value_losses.append(loss_info['value_loss'])
                    self.entropies.append(loss_info['entropy'])
                    self.total_losses.append(loss_info['total_loss'])
                    # Track mean reward for this training batch
                    self.mean_rewards.append(rewards.mean().item())
                
                # Clear buffer after training
                self.rollout_buffer.clear()
            
            if done:
                self.count_games += 1
                score = result.score
                self.env.reset()

                if record and self.count_games > self.config.min_deaths_to_record:
                    if self.config.value_for_end_game.value == ValueForEndGame.last_action.value:
                        self.steps += 1
                        self.recorded_actions.append(self.env.actions_length())
                        self._save_snapshot(self.steps)
                    elif self.config.value_for_end_game.value == ValueForEndGame.not_exist.value:
                        pass
                self._save_actions()

                if score > top_result:
                    top_result = score
                    self.save_agent(iteration)

                print(f'Game {self.count_games} | Score: {score} | Record: {top_result} | Iteration: {iteration} | Epsilon: {self.epsilon:.3f}')
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
        
        # Plot losses if we have any
        if self.policy_losses:
            print(f"\nPlotting {len(self.policy_losses)} training steps...")
            plot_losses(self.policy_losses, self.value_losses, self.entropies, self.total_losses, self.mean_rewards)

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
        """Save both actor and critic networks"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "count_games": self.count_games,
            "begin_iteration": iteration
        }, self.model_path)