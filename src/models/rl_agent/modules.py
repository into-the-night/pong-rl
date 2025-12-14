import os
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        h = self.linear1(x)
        h = self.dropout(h) 
        p = self.linear2(F.relu(h))
        p = F.softmax(p, dim=-1) + 1e-8
        return p.ravel()


class PolicyTrainer:
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
        
        
        # Use baseline subtraction instead of full standardization
        # This preserves reward magnitude better while reducing variance
        baseline = discounted_rewards.mean()
        discounted_rewards = discounted_rewards - baseline
        
        # Optional: mild scaling if std is very large
        if discounted_rewards.std() > 10.0:
            discounted_rewards = discounted_rewards / (discounted_rewards.std() + 1e-8)
        
        
        print(f"Mean log_prob: {losses.mean().item():.4f}")
        print(f"Raw rewards: Sum={rewards.sum().item():.2f}, Mean={rewards.mean().item():.2f}")
        print(f"Discounted rewards: Mean={discounted_rewards.mean().item():.4f}, Std={discounted_rewards.std().item():.4f}")

        # Scale losses by discounted rewards (policy gradient)
        policy_loss = torch.mul(losses, discounted_rewards).mul(-1).sum()
        
        # Add entropy bonus to encourage exploration
        if action_probs is not None:
            # Entropy = -sum(p * log(p)) for each action distribution
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
            total_loss = policy_loss - self.entropy_coef * entropy
            print(f"Policy Loss: {policy_loss.item():.4f} | Entropy: {entropy.item():.4f} | Total Loss: {total_loss.item():.4f}")
        else:
            total_loss = policy_loss
            print(f"Loss: {total_loss.item():.4f}")
        
        print("-" * 50)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()