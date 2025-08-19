import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import get_dataloader
from src.model import BaselineModel


class LitModel(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()
        self.model = BaselineModel(num_classes=num_classes)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds==y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds==y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', required=True)
    p.add_argument('--val-csv', required=True)
    p.add_argument('--root-dir', default='data/raw')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-epochs', type=int, default=10)
    args = p.parse_args()

    train_dl = get_dataloader(args.train_csv, args.root_dir, batch_size=args.batch_size, shuffle=True)
    val_dl = get_dataloader(args.val_csv, args.root_dir, batch_size=args.batch_size, shuffle=False)

    model = LitModel()
    logger = CSVLogger('logs', name='sprint1')
    ckpt = ModelCheckpoint(dirpath='models', filename='sprint1-{epoch}', save_top_k=1, monitor='val_loss', mode='min')

    # Use Lightning v2.0+ Trainer API: specify accelerator and devices
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1  # adjust if you want multi-GPU training
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=args.max_epochs, logger=logger, callbacks=[ckpt])

    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    main()
