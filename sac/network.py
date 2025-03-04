import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2
epsilon = 1e-8

class AWGNActionNoise(object): # Gaussian Noise
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x


def weight_init_(network):
    if isinstance(network, nn.Linear):
        torch.nn.init.kaiming_uniform(network)
        torch.nn.init.constant_(network.bias, 0)


# Use ValueNet to improve the stability of learning(optional)
class ValueNet(nn.Module):
    def __init__(self, state, hidden_dim):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(state, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.apply(weight_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Use double QNet to mitigate Q value
class QNet(nn.Module):
    def __init__(self, state, action, hidden_dim):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state + action, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.apply(weight_init_)

    def forward(self, state, action):
        x = F.relu(self.fc1(state + action))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Gaussian-like policy
class StochasticPolicyNet(nn.Module):
    def __init__(self, state, num_action, hidden_dim):
        super(StochasticPolicyNet, self).__init__()

        self.fc1 = nn.Linear(state, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, num_action)
        self.log_std = nn.Linear(hidden_dim, num_action)

        self.apply(weight_init_)

        self.noise = AWGNActionNoise(mu=np.zeros(num_action))  # Gaussian noise

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.std(x)
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
        log_prob = log_prob.sum(1, keepdim=True)
        # mu = torch.tanh(mu)

        return action, log_prob, mu