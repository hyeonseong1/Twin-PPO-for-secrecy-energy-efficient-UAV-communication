import torch
import random
import numpy as np

from environment.sumtree import SumTree
from environment.utils import device


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, state_size, action_size, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, state, action, reward, next_state, done):

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        # 전체 우선순위 합을 batch_size+1 구간으로 나눔
        segments = np.linspace(0, self.tree.total, batch_size + 1)
        # 각 구간에서 균등분포를 따르는 무작위 값을 추출
        cumsums = np.random.uniform(segments[:-1], segments[1:])

        # 각 cumsum 값에 대해 SumTree에서 샘플을 추출
        results = [self.tree.get(cumsum) for cumsum in cumsums]
        tree_idxs, priorities, sample_idxs = zip(*results)

        # 추출한 우선순위를 텐서로 변환 (배치 차원 유지)
        priorities = torch.tensor(priorities, dtype=torch.float).unsqueeze(1)

        # 샘플링 확률 및 중요도 가중치 계산
        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.state[list(sample_idxs)].to(device()),
            self.action[list(sample_idxs)].to(device()),
            self.reward[list(sample_idxs)].to(device()),
            self.next_state[list(sample_idxs)].to(device()),
            self.done[list(sample_idxs)].to(device())
        )
        return batch, weights, list(tree_idxs)


    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.state[sample_idxs].to(device()),
            self.action[sample_idxs].to(device()),
            self.reward[sample_idxs].to(device()),
            self.next_state[sample_idxs].to(device()),
            self.done[sample_idxs].to(device())
        )
        return batch