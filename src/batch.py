import torch
from torch.utils.data import Dataset

'''
One epoch is feeding in entire dataset once,
I want to find some lib that does so, so that I don't have to write logic from scratch.

'''
class BatchLoader(Dataset):
    def __init__(self, data, block_size):
        self.data = data  # Your encoded tensor
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # sliding window technique, cause we are basically predecting the next word and y is the actual label
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y