import random
import os
import math
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset


class NoisySineDataset(Dataset):
    def __init__(
        self,
        size: int = 1000, 
        max_chunks: int = 5,
        th_format = True
    ):

        self.size = size 
        self.max_chunks = max_chunks
        self.th_format = th_format
        
    def __len__(self):
        return self.size

    def get_noise(self, shape = 20, mean = 0.5):
        return np.random.normal(0, mean, shape)

    def get_sine(self, interval):
        shift_phase = random.randint(0,2)
        x = np.linspace((shift_phase+interval[0])*np.pi, (shift_phase+interval[1])*np.pi, 50)
        return np.sin(x)

    def __getitem__(self, i):

        #each chunk is either sin or noise
        num_chunks = random.randint(2,self.max_chunks)
        chunk_pos = [0 for i in range(num_chunks)]
        chunk_pos[random.sample(list(range(num_chunks)),1)[0]] = 1

        intervals = [[i*8, (i+1)* 8] for i in range(num_chunks)]
        
        x = []
        y = []

        for i in range(num_chunks):
            #if noise 
            if chunk_pos[i] == 0:
                x.append(np.linspace(intervals[i][0]*np.pi, intervals[i][1]*np.pi, 20))
                y.append(self.get_noise(shape=20))
            else:
                x.append(np.linspace(intervals[i][0]*np.pi, intervals[i][1]*np.pi, 50))
                y.append(self.get_sine(intervals[i]))

        np_x = np.concatenate(x)
        np_y = np.round(np.concatenate(y),3)

        torch_format = lambda x: torch.tensor(x, dtype=torch.float).view(-1, 1)
        if self.th_format:
            return torch_format(np_x), torch_format(np_y)
        
        return np_x, np_y

class CleanSineDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        th_format = True,
    ):
        self.size = size
        self.th_format = th_format

    def __len__(self):
        return self.size

    def get_sine(self, interval):
        shift_phase = random.randint(0,2)
        x = np.linspace((shift_phase+interval[0])*np.pi, (shift_phase+interval[1])*np.pi, 50)
        return np.sin(x)

    def __getitem__(self, i):
        num_chunks = random.randint(2,10)
        chunk_pos = [0 for i in range(num_chunks)]
        chunk_pos[random.sample(list(range(num_chunks)),1)[0]] = 1

        intervals = [[i*8, (i+1)* 8] for i in range(num_chunks)]
        
        x = []
        y = []

        for i in range(num_chunks):
            if chunk_pos[i] == 1:
                x.append(np.linspace(intervals[i][0]*np.pi, intervals[i][1]*np.pi, 50))
                y.append(self.get_sine(intervals[i]))

        np_x = np.concatenate(x)
        np_y = np.concatenate(y)

        torch_format = lambda x: torch.tensor(x, dtype=torch.float).view(-1, 1)
        if self.th_format:
            return torch_format(np_x), torch_format(np_y)
        
        return np_x, np_y


class ReverseDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        th_format = True,
    ):
        self.size = size
        self.th_format = th_format

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        correct = np.random.randint(low = 1, high = 10, size=random.randint(2, 20))
        correct = correct/10.0
        reverse = correct[::-1]
        
        torch_format = lambda x: torch.tensor(x.copy(), dtype=torch.float).view(-1, 1)
        if self.th_format:
            return torch_format(correct), torch_format(reverse)
        
        return correct, reverse


if __name__ == "__main__":
    k = NoisySineDataset()
    for i in range(10):
        next(iter(k))
    
    l = ReverseDataset()
    for i in range(10):
        c, r = next(iter(l))
        print(f"c {c}, r {r}")
