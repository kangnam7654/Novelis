import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_modules.data_module import DataModule
from models.regressor import Regressor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    return args


def main(args):
    model = Regressor()
    weight = torch.load(args.weight)
    model.load_state_dict(weight)
    model.eval()

    df = pd.read_csv(args.csv)

    dataset = DataModule(df, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    for data in loader:
        out = model(data)


if __name__ == "__main__":
    args = get_args()
    main(args)
