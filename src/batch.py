import torch

class GetBatch:

    def __init__(self, data, batch_size, block_size):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size

    def mini_batch(self):
        random_integers = torch.randint(0, len(self.data) - self.block_size, (self.batch_size,))
        input = torch.stack([self.data[curr:curr + self.block_size] for curr in random_integers])
        labels = torch.stack([self.data[curr+1:curr + self.block_size + 1] for curr in random_integers])
        return input, labels