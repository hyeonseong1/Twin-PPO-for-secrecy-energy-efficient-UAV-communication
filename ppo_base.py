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

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class Actor(nn.Module):
    def __init__(self,
                 alpha,
                 input_dim,
                 hidden1_dim,
                 hidden2_dim,
                 hidden3_dim,
                 n_action,  # action이 output
                 activation_f=torch.relu,
                 optimizer=optim.Adam):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim
        self.n_action = n_action
        self.activation = activation_f

        self.fc1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.bn1 = nn.LayerNorm(self.hidden1_dim)

        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.bn2 = nn.LayerNorm(self.hidden2_dim)

        self.fc3 = nn.Linear(self.hidden2_dim, self.hidden3_dim)
        self.bn3 = nn.LayerNorm(self.hidden3_dim)

        self.mu = nn.Linear(self.hidden3_dim, self.n_action)
        self.std = nn.Linear(self.hidden3_dim, self.n_action)

        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)

        mu = self.mu(x)
        log_std = self.std(x)
        std = torch.exp(log_std)

        return mu, std # UAV coordination(3) or RIS beamforming(20)

    # def save_checkpoint(self):
    #     print('... saving checkpoint ...')
    #     torch.save(self.state_dict(), self.checkpoint_file)
    #
    # def load_checkpoint(self, load_file=''):
    #     print('... loading checkpoint ...')
    #     if torch.cuda.is_available():
    #         self.load_state_dict(torch.load(load_file))
    #     else:
    #         self.load_state_dict(torch.load(load_file, map_location=torch.device('cpu')))

class Critic(nn.Module):
    def __init__(self,
                 beta,
                 input_dim,
                 hidden1_dim,
                 hidden2_dim,
                 hidden3_dim,
                 n_action,
                 activation_f=torch.relu,
                 optimizer=optim.Adam):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim
        self.n_action = n_action
        self.activation = activation_f

        self.fc1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.bn1 = nn.LayerNorm(self.hidden1_dim)

        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.bn2 = nn.LayerNorm(self.hidden2_dim)

        self.fc3 = nn.Linear(self.hidden2_dim, self.hidden3_dim)
        self.bn3 = nn.LayerNorm(self.hidden3_dim)

        self.v = nn.Linear(self.hidden3_dim, 1)

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

        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)
        state_value = self.activation(state_value)

        state_value = self.v(state_value)

        return state_value

    # def save_checkpoint(self):
    #     print('... saving checkpoint ...')
    #     torch.save(self.state_dict(), self.checkpoint_file)
    #
    # def load_checkpoint(self, load_file=''):
    #     print('... loading checkpoint ...')
    #     if torch.cuda.is_available():
    #         self.load_state_dict(torch.load(load_file))
    #     else:
    #         self.load_state_dict(torch.load(load_file, map_location=torch.device('cpu')))

class PPOAgent(object):
    def __init__(self,
                 alpha,
                 beta,
                 input_dim,
                 n_action,
                 lamda=1,
                 gamma=0.99,
                 epsilon=1e-7,
                 max_size=1e6,
                 layer1_size=400,
                 layer2_size=256,
                 layer3_size=128,
                 batch_size=64,
                 update_actor_interval=2,
                 noise='AWGN'):
        self.input_dim = input_dim
        self.lamda = lamda
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_action = n_action
        self.max_size = max_size
        self.batch_size = batch_size
        self.update_actor_inter = update_actor_interval
        self.noise = noise

        self.memory = ReplayBuffer(max_size, input_dim, n_action)

        # alpha for actor, beta for critic; learning rate hyperparameter
        self.actor = Actor(alpha, input_dim, layer1_size, layer2_size, layer3_size, n_action)
        self.critic = Critic(beta, input_dim, layer1_size, layer2_size, layer3_size, n_action)

        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_action))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu=np.zeros(n_action))

    def act(self, state, greedy=0.5):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)

        mu, std = self.actor(state)
        mu += torch.tensor(greedy * self.noise(), dtype=torch.float32).to(self.actor.device)
        act_prob = torch.normal(mu, std)
        action = torch.tanh(act_prob)
        self.actor.train()

        return action.detach().cpu().numpy()

    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        old_states = torch.tensor(states, dtype=torch.float32).to(self.actor.device)
        old_actions = torch.tensor(actions, dtype=torch.float32).to(self.actor.device)
        old_states_ = torch.tensor(states_, dtype=torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.actor.device)
        done_tensor = torch.ones_like(dones)

        # GAE 계산, old_policy 로그 확률 구하기
        with torch.no_grad():
            values = self.critic(old_states)
            next_values = self.critic(old_states_)

            deltas = rewards + (done_tensor-dones) * self.gamma * next_values - values
            advantages = torch.clone(deltas)
            for t in reversed(range(len(rewards) - 1)):
                advantages[t] += (1 - dones[t]) * self.gamma * self.lamda * advantages[t + 1]

            # Old policy's log probabilities (to calculate the ratio for PPO)
            mu_old, std_old = self.actor(old_states)
            old_log_probs = -0.5 * (
                        (old_actions - mu_old) ** 2 / (std_old ** 2) + torch.log(2 * np.pi * std_old ** 2))

        # Update actor and critic networks K epoch(update interval)
        for _ in range(self.update_actor_inter):
            # Calculate the ratio
            mu, std = self.actor(old_states)
            log_probs = -0.5 * ((old_actions - mu) ** 2 / (std ** 2) + torch.log(2 * np.pi * std ** 2))
            ratios = torch.exp(log_probs - old_log_probs)

            # Compute surrogate loss (PPO objective)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.actor.device)
            surr1 = ratios.T @ advantages_tensor
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon).T @ advantages_tensor
            kl_penalty = torch.mean(old_log_probs - log_probs)
            kl_penalty = 0.01 * kl_penalty
            actor_loss = torch.mean((-torch.min(surr1, surr2) + kl_penalty))

        # Update the actor network
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1000)
        self.actor.optimizer.step()

        # Update the critic network (value function)
        critic_target = rewards + self.gamma * next_values * (1 - dones)
        critic_loss = F.mse_loss(self.critic(old_states), critic_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1000)
        self.critic.optimizer.step()

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def reset(self):
        self.memory = ReplayBuffer(self.max_size, self.input_dim, self.n_action)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, load_file_actor = '',load_file_critic =''):
        self.actor.load_checkpoint(load_file = load_file_actor)
        self.critic.load_checkpoint(load_file = load_file_critic)

