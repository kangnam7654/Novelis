import pytorch_lightning as pl
import torch
import torch.nn as nn


class Pipeline(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        train_loss = self.loop(batch)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.loop(batch, stage="Valid")
        return valid_loss

    def loop(self, batch, stage="Train"):
        data, label = batch
        out = self.model(data)
        loss = self.criterion(out, label)

        predicted = (out > 0.5).float()
        accuracy = (predicted == label).float().mean()
        to_log = {f"{stage} Loss": loss, f"{stage} Accuracy": accuracy}
        self.log_dict(to_log, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
