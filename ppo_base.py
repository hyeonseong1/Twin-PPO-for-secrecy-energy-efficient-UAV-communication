import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class OUActionNoise(object): # 시간적 상관 관계
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class AWGNActionNoise(object): # 가우시안 상관관계
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        # self.mu = mu
        # self.sigma = sigma
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x

import numpy as np

class RolloutBuffer(object):
    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.buffer_cnt = 0

    def store_transition(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(state_)
        self.terminals.append(1 - done)
        self.buffer_cnt += 1

    def sample_buffer(self, batch_size):
        if self.buffer_cnt < batch_size:
            print(f"[Warning] Not enough samples in buffer: {self.buffer_cnt}/{batch_size}")
            return None  # 데이터 부족 시 None 반환

        batch = np.random.choice(self.buffer_cnt, batch_size)

        states = np.array(self.states, dtype=np.float32)[batch]
        actions = np.array(self.actions, dtype=np.float32)[batch]
        rewards = np.array(self.rewards, dtype=np.float32)[batch]
        next_states = np.array(self.next_states, dtype=np.float32)[batch]
        terminals = np.array(self.terminals, dtype=np.float32)[batch]

        return states, actions, rewards, next_states, terminals

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.terminals.clear()
        self.buffer_cnt = 0



class Actor(nn.Module):
    def __init__(self,
                 alpha,
                 input_dim,
                 hidden1_dim,
                 hidden2_dim,
                 n_action,  # action이 output
                 ):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.n_action = n_action

        self.fc1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.bn1 = nn.LayerNorm(self.hidden1_dim)

        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.bn2 = nn.LayerNorm(self.hidden2_dim)

        self.mu = nn.Linear(self.hidden2_dim, self.n_action)
        self.std = nn.Linear(self.hidden2_dim, self.n_action)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(self.bn1(x))

        x = self.fc2(x)
        x = torch.relu(self.bn2(x))

        mu = self.mu(x)
        log_std = self.std(x)
        std = torch.exp(log_std)

        return mu, std # UAV coordination(3) or RIS beamforming(20)


class Critic(nn.Module):
    def __init__(self,
                 beta,
                 input_dim,
                 hidden1_dim,
                 hidden2_dim,
                 n_action,
                 activation_f=torch.relu,
                 optimizer=optim.Adam):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.n_action = n_action
        self.activation = activation_f

        self.fc1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.bn1 = nn.LayerNorm(self.hidden1_dim)

        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.bn2 = nn.LayerNorm(self.hidden2_dim)

        self.v = nn.Linear(self.hidden2_dim, 1)

        self.optimizer = optimizer(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = self.activation(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = self.activation(state_value)

        state_value = self.v(state_value)

        return state_value


class PPOAgent(object):
    def __init__(self,
                 alpha,
                 beta,
                 input_dim,
                 n_action,
                 lamda=0.95,
                 gamma=0.99,
                 eps_clip=0.2,
                 layer1_size=400,
                 layer2_size=256,
                 layer3_size=128,
                 batch_size=64,
                 K_epochs=5,
                 noise='AWGN'):
        self.input_dim = input_dim
        self.lamda = lamda
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_action = n_action
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.noise_type = noise

        self.buffer = RolloutBuffer()

        # alpha for actor, beta for critic; learning rate hyperparameter
        self.actor = Actor(alpha, input_dim, layer1_size, layer2_size, n_action)

        self.critic = Critic(beta, input_dim, layer1_size, layer2_size, n_action)

        self.old_actor = copy.deepcopy(self.actor)

        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_action))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu=np.zeros(n_action))

    def act(self, state, greedy=0.5):
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)

        with torch.no_grad():
            mu, std = self.actor(state)
            std = torch.clamp(std, min=1e-6, max=2.0)

            # Apply noise to mean if exploration is desired
            if greedy > 0:
                noise_tensor = torch.tensor(greedy * self.noise(), dtype=torch.float32).to(self.actor.device)
                mu = mu + noise_tensor

            # Create distribution and sample
            try:
                dist = torch.distributions.Normal(mu, std)
                actions_raw = dist.sample()
                actions = torch.tanh(actions_raw)  # Bound actions to [-1, 1]
            except ValueError:
                # Fallback in case of invalid values
                print(f"Warning: Distribution error in act(). Using direct sampling. mu={mu}, std={std}")
                actions_raw = mu + torch.randn_like(mu) * std
                actions = torch.tanh(actions_raw)

        return actions.detach().cpu().numpy()

    def learn(self):
        # Get data from buffer
        sample = self.buffer.sample_buffer(self.batch_size)
        if sample is None:
            return  # Not enough data to train

        states, actions, rewards, states_, dones = sample

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.actor.device)
        next_states = torch.tensor(states_, dtype=torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.actor.device)
        done_tensor = torch.ones_like(dones)

        # Calculate advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

            # TD errors
            deltas = rewards + self.gamma * next_values * done_tensor - values

            # Calculate GAE
            advantages = torch.zeros_like(deltas)
            gae = 0
            for t in reversed(range(len(rewards))):
                if done_tensor[t]:
                    gae = deltas[t]
                else:
                    gae = deltas[t] + self.gamma * self.lamda * gae
                advantages[t] = gae

            # Calculate returns for critic update
            returns = advantages + values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # print(f'advantages_std: {advantages.std()}')

            # Get old policy distribution parameters
            mu_old, std_old = self.old_actor(states)

            # Calculate log probabilities of old actions
            try:
                old_dist = torch.distributions.Normal(mu_old, std_old)
                # Use atanh with numerical stability
                arctanh_actions = 0.5 * torch.log((1 + actions + 1e-7) / (1 - actions + 1e-7))
                old_log_probs = old_dist.log_prob(arctanh_actions).sum(1, keepdim=True)
            except ValueError as e:
                print(f"Error in old distribution: {e}")
                print(
                    f"mu_old stats: min={mu_old.min().item()}, max={mu_old.max().item()}, mean={mu_old.mean().item()}")
                print(
                    f"std_old stats: min={std_old.min().item()}, max={std_old.max().item()}, mean={std_old.mean().item()}")
                # Fallback calculation
                old_log_probs = -0.5 * ((arctanh_actions - mu_old) ** 2 / (std_old ** 2 + 1e-8)).sum(1, keepdim=True)
                old_log_probs = old_log_probs - 0.5 * torch.log(2 * np.pi * std_old ** 2).sum(1, keepdim=True)

        # PPO Update loop
        for _ in range(self.K_epochs):
            # Get current policy distribution parameters
            mu, std = self.actor(states)

            # Calculate log probabilities of actions under current policy
            try:
                current_dist = torch.distributions.Normal(mu, std)
                arctanh_actions = 0.5 * torch.log((1 + actions + 1e-7) / (1 - actions + 1e-7))
                current_log_probs = current_dist.log_prob(arctanh_actions).sum(1, keepdim=True)
            except ValueError as e:
                print(f"Error in current distribution: {e}")
                print(f"mu stats: min={mu.min().item()}, max={mu.max().item()}, mean={mu.mean().item()}")
                print(f"std stats: min={std.min().item()}, max={std.max().item()}, mean={std.mean().item()}")
                # Fallback calculation
                current_log_probs = -0.5 * ((arctanh_actions - mu) ** 2 / (std ** 2 + 1e-8)).sum(1, keepdim=True)
                current_log_probs = current_log_probs - 0.5 * torch.log(2 * np.pi * std ** 2).sum(1, keepdim=True)

            # Calculate policy ratio
            ratios = torch.exp(current_log_probs - old_log_probs)
            # print(f'ratios = {ratios}')
            # print(f'advantage: {advantages}')

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # print(f'surr1: {surr1}')
            # print(f'surr2: {surr2}')

            # Calculate actor loss with KL penalty
            try:
                kl_div = torch.distributions.kl_divergence(
                    torch.distributions.Normal(mu_old, std_old),
                    torch.distributions.Normal(mu, std)
                ).mean()
            except ValueError:
                # Fallback KL calculation if distribution fails
                kl_div = (torch.log(std / std_old) + (std_old ** 2 + (mu_old - mu) ** 2) / (2 * std ** 2) - 0.5).mean()
            # kl_div = ratios - torch.log(ratios) - 1

            # print(f'kl_div = {kl_div}')
            actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div
            # print(">>>actor_loss:", actor_loss.item())

            # Update actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100)
            self.actor.optimizer.step()

            # Calculate critic loss
            critic_value = self.critic(states)
            critic_loss = F.mse_loss(critic_value, returns)

            # Update critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100)
            self.critic.optimizer.step()

        # Update old actor for next iteration
        self.old_actor = copy.deepcopy(self.actor)

        # Clear buffer after update
        self.buffer.clear()

    def store_transition(self, state, action, reward, state_, done):
        self.buffer.store_transition(state, action, reward, state_, done)

