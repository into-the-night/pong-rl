from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

from models.rl_agent.modules import PolicyNetwork, ValueNetwork, PPOTrainer
from PongGame.env import PongEnv, make_vector_env


def plot_losses(policy_losses, value_losses, entropies, total_losses, mean_rewards):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        (policy_losses, 'Policy Loss', 'b-'),
        (value_losses, 'Value Loss', 'r-'),
        (entropies, 'Entropy', 'g-'),
        (total_losses, 'Total Loss', 'm-'),
        (mean_rewards, 'Mean Reward', 'orange'),
        (np.cumsum(mean_rewards), 'Cumulative Reward', 'cyan'),
    ]
    for ax, (data, title, style) in zip(axes.flat, metrics):
        ax.plot(data, style if isinstance(style, str) and len(style) <= 3 else 'b-', linewidth=1.5,
                color=style if not (isinstance(style, str) and len(style) <= 3) else None)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Update')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=200, bbox_inches='tight')
    print("Saved training_losses.png")
    plt.close(fig)


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
    epsilon_start: float = 0.0
    epsilon_min: float = 0.0
    epsilon_decay: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    train_every_iteration: int = 1
    save_every_iteration: Optional[int] = None


def _compute_gae(rewards, values, dones, next_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_v = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values
    return adv, returns


class PolicyAgent:
    def __init__(
        self,
        env: Optional[PongEnv],
        config: PolicyAgentConfig,
        model_path: str,
        dataset_path: str = "",
        last_checkpoint: Optional[str] = None,
    ):
        self.config = config
        self.model_path = model_path

        sample = env if env is not None else PongEnv()
        obs_dim = sample.observation_space.shape[0]
        n_actions = int(sample.action_space.n)

        self.actor = PolicyNetwork(obs_dim, config.hidden_state, n_actions)
        self.critic = ValueNetwork(obs_dim, config.hidden_state)
        self.trainer = PPOTrainer(
            actor=self.actor, critic=self.critic,
            lr=config.lr, gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon, value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef, max_grad_norm=config.max_grad_norm,
            ppo_epochs=config.ppo_epochs,
        )

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.count_games = 0
        self.begin_iteration = 0
        self.policy_losses, self.value_losses = [], []
        self.entropies, self.total_losses, self.mean_rewards = [], [], []

        if last_checkpoint:
            params = torch.load(last_checkpoint)
            self.actor.load_state_dict(params["actor"])
            self.critic.load_state_dict(params["critic"])
            self.trainer.optimizer.load_state_dict(params["optimizer"])
            self.count_games = params.get("count_games", 0)
            self.begin_iteration = params.get("begin_iteration", 0)

    def _act_batch(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.from_numpy(obs_np).float()
        with torch.no_grad():
            probs = self.actor(obs_t)
            values = self.critic(obs_t)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.numpy(), log_probs.numpy(), values.numpy()

    def train(
        self,
        num_envs: int = 5,
        rollout_steps: int = 256,
        show_plot: bool = False,
        max_games: Optional[int] = None,
        replay_best: bool = True,
        base_seed: int = 0,
    ):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

        venv = make_vector_env(num_envs, base_seed=base_seed)
        obs, _ = venv.reset(seed=base_seed)

        ep_rewards = np.zeros(num_envs, dtype=np.float64)
        ep_lengths = np.zeros(num_envs, dtype=np.int64)
        ep_states: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
        best_reward = -float("inf")
        best_states: List[np.ndarray] = []
        best_game_idx = -1

        print(f"Vectorized PPO: num_envs={num_envs}, rollout_steps={rollout_steps}")
        if max_games:
            print(f"Stopping after {max_games} games")

        iteration = self.begin_iteration
        total_iterations = self.config.iterations
        stop = False

        while iteration < total_iterations and not stop:
            T = rollout_steps
            buf_states = np.zeros((T, num_envs, self.obs_dim), dtype=np.float32)
            buf_actions = np.zeros((T, num_envs), dtype=np.int64)
            buf_logp = np.zeros((T, num_envs), dtype=np.float32)
            buf_values = np.zeros((T, num_envs), dtype=np.float32)
            buf_rewards = np.zeros((T, num_envs), dtype=np.float32)
            buf_dones = np.zeros((T, num_envs), dtype=np.float32)

            for t in range(T):
                actions, log_probs, values = self._act_batch(obs)
                buf_states[t] = obs
                buf_actions[t] = actions
                buf_logp[t] = log_probs
                buf_values[t] = values

                next_obs, rewards, terms, truncs, _infos = venv.step(actions)
                dones = np.logical_or(terms, truncs)

                buf_rewards[t] = rewards
                buf_dones[t] = dones.astype(np.float32)

                for i in range(num_envs):
                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1
                    ep_states[i].append(obs[i].copy())
                    if dones[i]:
                        self.count_games += 1
                        if ep_rewards[i] > best_reward:
                            best_reward = float(ep_rewards[i])
                            best_states = ep_states[i].copy()
                            best_game_idx = self.count_games
                            self.save_agent(iteration)
                        print(f"Game {self.count_games} | reward={ep_rewards[i]:7.2f} | "
                              f"len={ep_lengths[i]:4d} | best={best_reward:7.2f} "
                              f"(game {best_game_idx}) | env={i}")
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0
                        ep_states[i] = []
                        if max_games and self.count_games >= max_games:
                            stop = True

                obs = next_obs
                if stop:
                    break

            steps_collected = (t + 1) if stop else T
            with torch.no_grad():
                next_values = self.critic(torch.from_numpy(obs).float()).numpy()

            advantages = np.zeros((steps_collected, num_envs), dtype=np.float32)
            returns = np.zeros((steps_collected, num_envs), dtype=np.float32)
            for e in range(num_envs):
                adv, ret = _compute_gae(
                    buf_rewards[:steps_collected, e],
                    buf_values[:steps_collected, e],
                    buf_dones[:steps_collected, e],
                    float(next_values[e]),
                    self.config.gamma, self.config.gae_lambda,
                )
                advantages[:, e] = adv
                returns[:, e] = ret

            flat = lambda arr: arr.reshape(-1, *arr.shape[2:])
            loss_info = self.trainer.update(
                states=torch.from_numpy(flat(buf_states[:steps_collected])).float(),
                actions=torch.from_numpy(flat(buf_actions[:steps_collected])).long(),
                old_log_probs=torch.from_numpy(flat(buf_logp[:steps_collected])).float(),
                advantages=torch.from_numpy(flat(advantages)).float(),
                returns=torch.from_numpy(flat(returns)).float(),
                old_values=torch.from_numpy(flat(buf_values[:steps_collected])).float(),
            )
            if loss_info:
                self.policy_losses.append(loss_info['policy_loss'])
                self.value_losses.append(loss_info['value_loss'])
                self.entropies.append(loss_info['entropy'])
                self.total_losses.append(loss_info['total_loss'])
                self.mean_rewards.append(float(buf_rewards[:steps_collected].mean()))
                print(f"[update] iter={iteration} pol={loss_info['policy_loss']:.4f} "
                      f"val={loss_info['value_loss']:.4f} ent={loss_info['entropy']:.4f}")

            iteration += steps_collected
            if (self.config.save_every_iteration is not None
                    and iteration % self.config.save_every_iteration < steps_collected):
                self.save_agent(iteration)

        venv.close()
        self.save_agent(iteration)
        print(f"\nFinished. Total games: {self.count_games}. Best reward: {best_reward:.2f}")

        if self.policy_losses and show_plot:
            plot_losses(self.policy_losses, self.value_losses, self.entropies,
                        self.total_losses, self.mean_rewards)

        if replay_best and best_states:
            replay_episode(best_states, best_reward, best_game_idx)

    def save_agent(self, iteration: int):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "count_games": self.count_games,
            "begin_iteration": iteration,
        }, self.model_path)


def replay_episode(states: List[np.ndarray], total_reward: float, game_idx: int,
                   save_path: Optional[str] = "best_episode.gif"):
    """Animate the best episode and optionally save as a gif."""
    import matplotlib.animation as animation
    from PongGame.game import render_from_state

    print(f"\nReplaying best episode (game {game_idx}, reward={total_reward:.2f}, "
          f"{len(states)} frames)...")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(render_from_state(states[0]))
    ax.set_title(f"Best episode — game {game_idx} — reward {total_reward:.2f}")
    ax.axis('off')

    def update(i):
        im.set_data(render_from_state(states[i]))
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(states),
                                   interval=16, blit=True, repeat=False)
    if save_path:
        try:
            anim.save(save_path, writer=animation.PillowWriter(fps=60))
            print(f"Saved replay to {save_path}")
        except Exception as e:
            print(f"Could not save gif ({e}); showing interactively instead.")
    plt.show()
