import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sac.network import *
from sac.replay_buffer import ReplayBuffer

class SAC:
    def __init__(self,
                 state_dim,
                 n_action,
                 gamma,     # discount factor
                 tau,       # soft update factor
                 alpha,     # entropy temperature
                 total_episodes,
                 hidden_dim,
                 learning_rate,
                 max_size,
                 batch_size):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.initial_alpha = alpha
        self.final_alpha = 0.01
        self.total_episodes = total_episodes  # 전체 에피소드 수
        self.current_episode = 0  # 현재 에피소드 진행 상황

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q1 = QNet(state_dim, n_action, hidden_dim).to(self.device)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=learning_rate)
        self.Q1_target = copy.deepcopy(self.Q1)

        self.Q2 = QNet(state_dim, n_action, hidden_dim).to(self.device)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=learning_rate)
        self.Q2_target = copy.deepcopy(self.Q2)

        self.policy = StochasticPolicyNet(state_dim, n_action, hidden_dim).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.max_size = max_size
        self.buffer = ReplayBuffer(self.max_size)

        self.batch_size = batch_size

    def update_alpha(self):
        # 선형 감소
        progress = self.current_episode / self.total_episodes
        self.alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress

        # # 지수 감소 (선택 가능)
        # decay_rate = -np.log(self.final_alpha / self.initial_alpha) / self.total_episodes
        # self.alpha = self.initial_alpha * np.exp(-decay_rate * self.current_episode)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, _, _ = self.policy.sample(state)

        return action.detach().cpu().numpy()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            print("Not enough buffer...")
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.buffer.sample(self.batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device).unsqueeze(1)

        with torch.no_grad():
            action_prime, log_pi_prime, _ = self.policy.sample(next_state_batch)   # action, log_pi, mu
            q1_target = self.Q1_target(next_state_batch, action_prime)
            q2_target = self.Q2_target(next_state_batch, action_prime)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * log_pi_prime  # Soft Value
            next_q_value = reward_batch + self.gamma * min_q_target * (1 - done_batch)

        q1 = self.Q1(state_batch, action_batch)
        q2 = self.Q2(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, next_q_value).mean()
        q2_loss = F.mse_loss(q2, next_q_value).mean()


        self.Q1_optim.zero_grad()
        q1_loss.backward()
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        q2_loss.backward()
        self.Q2_optim.step()

        # re-calculate Q value for update policy
        pi, log_pi, _ = self.policy.sample(state_batch)
        q1_pi = self.Q1(state_batch, pi)
        q2_pi = self.Q2(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_pi - min_q_pi).mean()


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # print(f'    Q1_loss = {q1_loss}, Q2_loss = {q2_loss}, policy_loss = {policy_loss}')

        # Soft update
        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)