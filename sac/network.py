import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2
epsilon = 1e-8

class AWGNActionNoise(object): # 가우시안 상관관계
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x


def he_initialization(layer, is_output=False):
    if isinstance(layer, nn.Linear):
        if is_output:
            nn.init.uniform_(layer.weight, -1e-3, 1e-3)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        else:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

# (optional)
# Use ValueNet to improve the stability of learning
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # he_initialization(self.fc1)
        # he_initialization(self.fc2)
        # he_initialization(self.fc3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Use double QNet to mitigate Q value
class QNet(nn.Module):
    def __init__(self, state_dim, n_action, hidden_dim):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim + n_action, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # he_initialization(self.fc1)
        # he_initialization(self.fc2)
        # he_initialization(self.fc3)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)

        return x


# Gaussian-like policy
class StochasticPolicyNet(nn.Module):
    def __init__(self, state_dim, n_action, hidden_dim, action_scale=1.0):
        super(StochasticPolicyNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, n_action)
        self.log_std = nn.Linear(hidden_dim, n_action)

        # he_initialization(self.fc1)
        # he_initialization(self.fc2)
        # he_initialization(self.mu)
        # he_initialization(self.log_std)

        self.action_scale = action_scale

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)

        log_std = self.log_std(x)
        log_std = torch.clip(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX) # mitigate gradient exploding
        std = torch.exp(log_std)

        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = dist.log_prob(x_t)
        # 행동을 -1 ~ 1로 제한: 확률 밀도를 고려 하여 행동 로그 확률 재조정
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        return action, log_prob, mu