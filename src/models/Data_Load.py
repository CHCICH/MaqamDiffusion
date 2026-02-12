import torch
import torch.nn as nn

class Dataset_(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class DataLoader_AutoEncoder:
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            torch.random.shuffle(indices)

        batches = []
        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = torch.stack([torch.tensor(self.dataset[i]) for i in batch_indices])
            batches.append(batch)
        
        return batches


class DataLoader_Diffusion:
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            torch.random.shuffle(indices)

        batches = []
        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = torch.stack([torch.tensor(self.dataset[i]) for i in batch_indices])
            batches.append(batch)
        
        return batches

