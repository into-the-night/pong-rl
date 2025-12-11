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
    def __init__(self, model: torch.nn.Module, lr: float, alpha: float, gamma: float):
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)

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
        dones: Union[List[bool], torch.Tensor]
    ):
        self.optimizer.zero_grad()
        discounted_rewards = self.get_rewards(rewards, dones)
        if not isinstance(discounted_rewards, torch.Tensor):
            discounted_rewards = torch.tensor(discounted_rewards, dtype=losses.dtype)
        
        # Normalize rewards
        if len(discounted_rewards) > 1 and discounted_rewards.std() > 1e-8:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        else:
            # If variance is 0 (e.g. all games identical), don't normalize to 0. 
            # Just center them or leave raw. Leaving raw is safer for now to maintain sign.
            print("Warning: Reward std dev is 0. Skipping normalization.")
        
        print(f"Mean log_prob: {losses.mean().item():.4f}")
        print(f"Rewards stats: Mean={discounted_rewards.mean().item():.4f}, Std={discounted_rewards.std().item():.4f}")

        # Scale losses by discounted rewards
        losses = torch.mul(losses, discounted_rewards).mul(-1)
        
        loss = torch.sum(losses)
        print(f"This is loss: {loss}")
        print(f"This is reward: {rewards.sum()}")
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()