import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sac.network import *
from sac.replay_buffer import Replaybuffer

class SAC:
    def __init__(self, state, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha     # entropy temperature

        self.policy = args.policy
        self.target_update_interval = args.target_update_interval
        self.entropy_rate = args.entropy_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q1 = QNet(state, action_space.shape[0], args.hidden_size).to(self.device)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q1_optim = optim.Adam(self.critic1.parameters(), lr=args.lr)

        self.Q2 = QNet(state, action_space.shape[0], args.hidden_size).to(self.device)
        self.Q2_target = copy.deepcopy(self.Q2)
        self.Q2_optim = optim.Adam(self.critic2.parameters(), lr=args.lr)

        self.policy = StochasticPolicyNet(state, action_space, args.hidden_dim)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.policy(state)
        return action.detach().cpu().numpy()[0]

    def learn(self, buffer, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = Replaybuffer.sample(batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_prime, action_log_pi, _ = self.policy.sample(next_state_batch)   # action, log_pi, mu
            q1_target = self.Q1_target(next_state_batch, action_prime)
            q2_target = self.Q2_target(next_state_batch, action_prime)
            min_q = torch.min(q1_target, q2_target)
            next_q_value = reward_batch + self.gamma * min_q
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
        q1 = self.Q1(state_batch, pi)
        q2 = self.Q2(state_batch, pi)
        min_q = torch.min(q1, q2)

        policy_loss = (self.alpha * log_pi - min_q).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Soft update
        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


