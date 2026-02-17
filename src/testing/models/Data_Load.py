import torch
import torch.nn as nn


def normalize_data(data):
    normalized_data = []
    min_val = -80
    max_val = 0
    for tensor in data:
        if max_val > min_val:  
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            normalized_tensor = tensor  
        normalized_data.append(normalized_tensor)
    return normalized_data

class Dataset_(torch.utils.data.Dataset):
    def __init__(self, data,normalize=False):
        self.data = data
        self.normalize = normalize
        if self.normalize:
            self.data = normalize_data(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]          
        x = x.unsqueeze(0)          
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

