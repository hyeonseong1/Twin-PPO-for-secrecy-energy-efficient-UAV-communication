# sac/prioritized_replay_buffer.py
import torch
import numpy as np
from sac.sumtree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, state_size, action_size, device, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps
        self.device = device  # SAC에서 전달받은 device 사용
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, self.count)
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        segments = np.linspace(0, self.tree.total, batch_size + 1)
        cumsums = np.random.uniform(segments[:-1], segments[1:])
        results = [self.tree.get(cumsum) for cumsum in cumsums]
        tree_idxs, priorities, sample_idxs = zip(*results)
        priorities = torch.tensor(priorities, dtype=torch.float).unsqueeze(1)
        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.state[list(sample_idxs)].to(self.device),
            self.action[list(sample_idxs)].to(self.device),
            self.reward[list(sample_idxs)].to(self.device),
            self.next_state[list(sample_idxs)].to(self.device),
            self.done[list(sample_idxs)].to(self.device)
        )
        return batch, weights.to(self.device), list(tree_idxs)

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (abs(priority) + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)