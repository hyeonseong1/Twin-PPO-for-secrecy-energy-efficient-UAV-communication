import random
import numpy as np

class Replaybuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) > self.max_size:
            assert "memory is full!"
        episode = (state, action, reward, next_state, done)
        self.buffer.append(episode)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


