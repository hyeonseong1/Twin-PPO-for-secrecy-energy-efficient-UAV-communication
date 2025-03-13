# sac/model_per.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sac.network import *
from sac.prioritized_replay_buffer import PrioritizedReplayBuffer

class SAC:
    def __init__(self,
                 state_dim,
                 n_action,
                 gamma,
                 tau,
                 alpha,
                 hidden_dim,
                 learning_rate,
                 hidden_size,
                 max_size,
                 batch_size):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q1 = QNet(state_dim, n_action, hidden_dim).to(self.device)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=learning_rate)
        self.Q1_target = copy.deepcopy(self.Q1).to(self.device)

        self.Q2 = QNet(state_dim, n_action, hidden_dim).to(self.device)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=learning_rate)
        self.Q2_target = copy.deepcopy(self.Q2).to(self.device)

        self.policy = StochasticPolicyNet(state_dim, n_action, hidden_size).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.buffer = PrioritizedReplayBuffer(
            buffer_size=max_size,
            state_size=state_dim,
            action_size=n_action,
            device=self.device,  # SAC의 self.device 전달
            eps=1e-2,
            alpha=0.6,
            beta=0.4
        )
        self.max_size = max_size
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()

    def learn(self):
        if self.buffer.real_size < self.batch_size:
            return

        (state_batch, action_batch, reward_batch, next_state_batch, done_batch), weights, tree_idxs = \
            self.buffer.sample(self.batch_size)

        reward_batch = reward_batch.unsqueeze(1)  # (512, 1)
        done_batch = done_batch.unsqueeze(1).float()  # (512, 1)
        weights = weights.to(self.device)  # (512, 1)

        # Compute target Q-value
        with torch.no_grad():
            action_prime, action_log_pi, _ = self.policy.sample(next_state_batch)
            q1_target = self.Q1_target(next_state_batch, action_prime)
            q2_target = self.Q2_target(next_state_batch, action_prime)
            min_q = torch.min(q1_target, q2_target)
            min_q = min_q.mean(dim=1, keepdim=True)  # Ensure (512, 1)
            action_log_pi = action_log_pi.mean(dim=1, keepdim=True)  # Ensure (512, 1)
            next_q_value = reward_batch + self.gamma * (1 - done_batch) * (min_q - self.alpha * action_log_pi)

        # Q-network losses
        q1 = self.Q1(state_batch, action_batch)  # (512, 1)
        q2 = self.Q2(state_batch, action_batch)  # (512, 1)
        q1_loss = (weights * F.mse_loss(q1, next_q_value, reduction='none')).mean()
        q2_loss = (weights * F.mse_loss(q2, next_q_value, reduction='none')).mean()

        # Compute TD errors for PER
        with torch.no_grad():
            td_errors = torch.abs(q1 - next_q_value).squeeze()  # (512,)

        # Update Q-networks
        self.Q1_optim.zero_grad()
        q1_loss.backward()
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        q2_loss.backward()
        self.Q2_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        q1_pi = self.Q1(state_batch, pi)
        q2_pi = self.Q2(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (weights * (self.alpha * log_pi - min_q_pi)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.buffer.update_priorities(tree_idxs, td_errors)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)