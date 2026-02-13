import torch
import torch.nn as nn


class Dataset_(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]          # shape [128, 6000]
        x = x.unsqueeze(0)          # add channel → [1, 128, 6000]
        return x


class DataLoader_AutoEncoder:
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start+self.batch_size]
            batch = torch.stack([self.dataset[i] for i in batch_indices])
            yield batch



class DataLoader_Diffusion:
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = torch.stack([torch.tensor(self.dataset[i]) for i in batch_indices])
            yield batch

