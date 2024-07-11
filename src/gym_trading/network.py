import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, inputs_features, n_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(inputs_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(inputs_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

    
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = torch.distributions.Categorical(probs)

        return value, dist
    

    def set_params(self, lr=0.01, lr_decay=0.995):
        optmizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optmizer, gamma=lr_decay)

        return optmizer, scheduler
