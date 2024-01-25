import argparse
from tqdm import tqdm 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_modules.data_module import DataModule
from models.regressor import Regressor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
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

    predictions = []
    for data in tqdm(loader):
        logit = model(data)
        if logit.item() >= 0.5:
            out = 1
        else:
            out = 0
        predictions.append(out)
    df["class"] = predictions
    df.to_csv("./data/prediction.csv")
    pass


if __name__ == "__main__":
    args = get_args()
    args.weight = "/home/kangnam/project/Novelis/weights/regressor.pt"
    args.csv = "/home/kangnam/project/Novelis/data/test.csv"
    main(args)
