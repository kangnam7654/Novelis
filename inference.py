import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_modules.data_module import DataModule
from models.regressor import Regressor

def main():
    model = Regressor()
    weight = torch.load("./weights/regressor.pt")
    model.load_state_dict(weight)
    model.eval()
    
    df = pd.read_csv("./data/test.csv")
    
    dataset = DataModule(df, train=False)
    loader = DataLoader(dataset, batch_size=1)
    
    for data in loader:
        out = model(data)

if __name__ == "__main__":
    main()