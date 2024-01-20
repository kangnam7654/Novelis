import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data_modules.data_module import DataModule
from pipelines.pipeline import Pipeline
from models.regressor import Regressor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--file_name", type=str, default="train.csv")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def main(args):
    # =============
    # | Load Data |
    # =============
    DATA_DIR = args.data_dir
    file_name = args.file_name
    df = pd.read_csv(Path(DATA_DIR).joinpath(file_name))

    # Split data for validation. To keep class balance, split fraud, non-fraud cases first.
    fraud = df[df["class"] == 1]  # Fraud Cases
    non_fraud = df[df["class"] == 0]  # Non-Faurd Cases
    train1, valid1 = train_test_split(fraud, test_size=0.2)  # Use 20% for Validation
    train2, valid2 = train_test_split(
        non_fraud, test_size=0.2
    )  # Use 20% for Validation

    train = pd.concat([train1, train2]).reset_index(drop=True)
    valid = pd.concat([valid1, valid2]).reset_index(drop=True)

    train_dataset = DataModule(train)
    valid_dataset = DataModule(valid)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # =======================
    # | Logger  & Callbacks |
    # =======================
    logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/", monitor="Valid Loss", verbose=True
    )

    # =========================
    # | Load Model & Pipeline |
    # =========================
    model = Regressor(input_dim=7)
    pipeline = Pipeline(model=model, lr=args.lr)

    # ========================
    # | Load Trainer & Train |
    # ========================
    trainer = pl.Trainer(
        logger=logger,
        callbacks=checkpoint_callback,
    )
    trainer.fit(
        model=pipeline, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
