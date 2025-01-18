import os
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=740, out_channels=10, kernel_size=32, bias=False)
        nn.init.xavier_uniform_(self.conv1d.weight)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x[60:,20:]
        x = x.unsqueeze(0)
        h = self.conv1d(x)
        h = h.view(h.size(0), -1)
        logp = self.linear2(F.relu(h))
        p = F.sigmoid(logp)
        return p.ravel()


class PolicyTrainer:
    def __init__(self, model: torch.nn.Module, lr: float, alpha: float, gamma: float):
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)

    def get_rewards(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, rewards.size(dim=0))):
            if rewards[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def train_step(
        self,
        state: torch.Tensor,
        losses: Union[torch.Tensor, List[int]],
        rewards: Union[torch.Tensor, float],
        done: Union[torch.Tensor, bool]
    ):
        self.optimizer.zero_grad()
        discounted_rewards = self.get_rewards(rewards)
        if not isinstance(discounted_rewards, torch.Tensor):
            discounted_rewards = torch.tensor(discounted_rewards, dtype=losses.dtype)
        # Scale losses by discounted rewards
        losses = losses * discounted_rewards
        # Normalize losses
        losses = (losses - losses.mean()) / (losses.std() + 1e-8)
        loss = torch.sum(losses)
        # if isinstance(losses, torch.Tensor):
        #     losses = losses.clone().detach().requires_grad_(True)
        # mean_loss = losses.mean()
        print(f"This is loss: {loss}")
        loss.backward()
        self.optimizer.step()