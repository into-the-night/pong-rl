import os
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """Actor network for PPO - outputs action probabilities"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        nn.init.orthogonal_(self.linear1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.linear1.bias, 0.0)
        
        self.linear2 = nn.Linear(hidden_size, output_size)
        nn.init.orthogonal_(self.linear2.weight, gain=0.01)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        h = torch.tanh(self.linear1(x))
        logits = self.linear2(h)
        probs = F.softmax(logits, dim=-1)
        return probs


class ValueNetwork(nn.Module):
    """Critic network for PPO - outputs state value estimate"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        nn.init.orthogonal_(self.linear1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.linear1.bias, 0.0)
        
        self.linear2 = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.linear2.weight, gain=1.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        h = torch.tanh(self.linear1(x))
        value = self.linear2(h)
        return value.squeeze(-1)


class PPOTrainer:
    """PPO Trainer with clipped surrogate objective and GAE"""
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        
        # Use Adam optimizer for both networks
        self.mini_batch_size = mini_batch_size
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=lr
        )

    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Tensor of rewards [T]
            values: Tensor of value estimates [T]
            dones: Tensor of done flags [T]
            next_value: Value estimate for next state after trajectory
            
        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: float = 0.0
    ):
        """
        Perform PPO training step with multiple epochs
        
        Args:
            states: Batch of states [T, state_dim]
            actions: Batch of actions [T]
            old_log_probs: Log probs from behavior policy [T]
            rewards: Batch of rewards [T]
            dones: Batch of done flags [T]
            values: Value estimates from behavior policy [T]
            next_value: Value estimate for next state
            
        Returns:
            Dictionary containing loss information
        """
        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Store raw advantages for logging
        raw_adv_mean = advantages.mean().item()
        raw_adv_std = advantages.std().item()
        
        # Normalize advantages only if std is not too small
        if advantages.std() > 1e-4:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # If advantages are too small, don't normalize (might indicate a problem)
            print(f"WARNING: Advantage std too small ({advantages.std().item():.6f}), skipping normalization")
        
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        
        # Detach old values for clipping
        old_values = values.detach()
        
        # Store losses from first epoch for return
        loss_info = {}
        
        # PPO update for multiple epochs
        for epoch in range(self.ppo_epochs):
            # Get current policy distribution
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Get current value estimates
            new_values = self.critic(states)
            
            # Compute probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss with clipping (stabilizes training)
            # Clip the value function updates similar to policy
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.clip_epsilon,
                self.clip_epsilon
            )
            value_losses = (new_values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Store losses from first epoch
            if epoch == 0:
                loss_info = {
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                    'total_loss': loss.item()
                }
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                self.max_grad_norm
            )
            self.optimizer.step()
            
            # Log training metrics
            if epoch == 0:
                approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                clipped_frac = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean().item()
                
                print(f"PPO Epoch {epoch+1}/{self.ppo_epochs}")
                print(f"Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
                print(f"Entropy: {entropy.item():.4f} | Total Loss: {loss.item():.4f}")
                print(f"Ratio: Mean={ratio.mean().item():.4f}, Min={ratio.min().item():.4f}, Max={ratio.max().item():.4f}")
                print(f"Approx KL: {approx_kl:.4f} | Clipped Fraction: {clipped_frac:.4f}")
                print(f"Raw Advantage: Mean={raw_adv_mean:.4f}, Std={raw_adv_std:.4f}")
                print(f"Norm Advantage: Mean={advantages.mean().item():.4f}, Std={advantages.std().item():.4f}")
                print(f"Returns: Mean={returns.mean().item():.4f}, Std={returns.std().item():.4f}")
                print(f"Rewards: Mean={rewards.mean().item():.4f}, Sum={rewards.sum().item():.4f}")
                print("-" * 60)
        
        return loss_info

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ):
        """PPO update with precomputed advantages and returns (for vectorized rollouts)."""
        raw_adv_mean = advantages.mean().item()
        raw_adv_std = advantages.std().item()
        if advantages.std() > 1e-4:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        mb = self.mini_batch_size
        loss_info = {}

        for epoch in range(self.ppo_epochs):
            idx = torch.randperm(n)
            epoch_pol, epoch_val, epoch_ent, epoch_tot = 0.0, 0.0, 0.0, 0.0
            num_batches = 0

            for start in range(0, n, mb):
                b = idx[start:start + mb]
                s = states[b]
                a = actions[b]
                olp = old_log_probs[b]
                adv = advantages[b]
                ret = returns[b]
                ov = old_values[b]

                action_probs = self.actor(s)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(a)
                entropy = dist.entropy().mean()
                new_values = self.critic(s)

                ratio = torch.exp(new_log_probs - olp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = ov + torch.clamp(new_values - ov, -self.clip_epsilon, self.clip_epsilon)
                v_loss_unclipped = (new_values - ret) ** 2
                v_loss_clipped = (value_pred_clipped - ret) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                epoch_pol += policy_loss.item()
                epoch_val += value_loss.item()
                epoch_ent += entropy.item()
                epoch_tot += loss.item()
                num_batches += 1

            if epoch == 0:
                loss_info = {
                    'policy_loss': epoch_pol / num_batches,
                    'value_loss': epoch_val / num_batches,
                    'entropy': epoch_ent / num_batches,
                    'total_loss': epoch_tot / num_batches,
                    'raw_adv_mean': raw_adv_mean,
                    'raw_adv_std': raw_adv_std,
                }

        return loss_info


# Backward compatibility - keep PolicyTrainer for reference
class PolicyTrainer:
    """Legacy vanilla policy gradient trainer - kept for reference"""
    def __init__(self, model: torch.nn.Module, lr: float, alpha: float, gamma: float, entropy_coef: float = 0.01):
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
        self.entropy_coef = entropy_coef

    def get_rewards(self, rewards, dones):
        discounted_r = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, rewards.size(dim=0))):
            if dones[t]: running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def train_step(
        self,
        state: torch.Tensor,
        losses: Union[torch.Tensor, List[int]],
        rewards: Union[torch.Tensor, float],
        dones: Union[List[bool], torch.Tensor],
        action_probs: torch.Tensor = None
    ):
        self.optimizer.zero_grad()
        discounted_rewards = self.get_rewards(rewards, dones)
        if not isinstance(discounted_rewards, torch.Tensor):
            discounted_rewards = torch.tensor(discounted_rewards, dtype=losses.dtype)
        
        baseline = discounted_rewards.mean()
        discounted_rewards = discounted_rewards - baseline
        
        if discounted_rewards.std() > 10.0:
            discounted_rewards = discounted_rewards / (discounted_rewards.std() + 1e-8)
        
        policy_loss = torch.mul(losses, discounted_rewards).mul(-1).sum()
        
        if action_probs is not None:
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
            total_loss = policy_loss - self.entropy_coef * entropy
        else:
            total_loss = policy_loss
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()