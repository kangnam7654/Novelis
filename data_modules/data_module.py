import pandas as pd
import torch
from torch.utils.data import Dataset


class DataModule(Dataset):
    def __init__(self, df, train=True):
        super().__init__()
        self.df = df
        self.train = train
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.train:
            data = self.df.iloc[idx, :-1]
            label = self.df.iloc[idx, -1]
            data = data.to_numpy()
            data = torch.from_numpy(data).float()
            label = torch.tensor(label).float().unsqueeze(0)
            return data, label
        else:
            data = self.df.iloc[idx]
            data = data.to_numpy()
            data = torch.from_numpy(data).float()
            return data

    def pre_process(self, data):
        return data
