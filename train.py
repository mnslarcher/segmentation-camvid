import argparse

import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
from torchmetrics import IoU
from torchmetrics.functional import dice_score

from segmentation.data import SegmentationDataModule
from segmentation.utils import object_from_dict


def _parse_args():
    parser = argparse.ArgumentParser("Training.")
    parser.add_argument(
        "--config-path", type=str, required=True, help="path to the configuration file."
    )
    return parser.parse_known_args()


class SegmentationModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = object_from_dict(hparams["model"])
        self.loss = object_from_dict(hparams["loss"])
        self.iou = IoU(hparams["model"]["classes"])
        self.dice = dice_score

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        preds = torch.softmax(out, dim=1)
        self.log("loss", loss)
        return {"loss": loss, "preds": preds}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        preds = torch.softmax(out, dim=1)
        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):
        self.log("val_loss", outputs["loss"])
        self.log("val_iou", self.iou(outputs["preds"], outputs["target"]))
        self.log("val_dice", self.dice(outputs["preds"], outputs["target"]))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True)
        self.log(
            "test_iou",
            self.iou(outputs["preds"], outputs["target"]),
        )
        self.log(
            "test_dice",
            dice_score(outputs["preds"], outputs["target"]),
        )

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=self.parameters()
        )
        scheduler = object_from_dict(
            self.hparams["scheduler"]["scheduler"],
            optimizer=optimizer,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": self.hparams["scheduler"]["interval"],
            "frequency": self.hparams["scheduler"]["frequency"],
        }
        return [optimizer], [scheduler_dict]


if __name__ == "__main__":
    args, _ = _parse_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams["model"]["model"]["classes"] = len(hparams["categories"])
    pl.seed_everything(hparams["seed"])
    hparams["data"]["data"]["transforms"] = {
        "train": from_dict(hparams["data"]["transforms"]["train"]),
        "val": from_dict(hparams["data"]["transforms"]["val"]),
        "test": from_dict(hparams["data"]["transforms"]["test"]),
    }
    data = SegmentationDataModule(**hparams["data"]["data"])
    model = SegmentationModule(hparams["model"])
    trainer = object_from_dict(
        hparams["trainer"]["trainer"],
        logger=object_from_dict(hparams["trainer"]["logger"]),
        callbacks=[
            object_from_dict(callback)
            for callback in hparams["trainer"]["callbacks"].values()
        ],
    )
    trainer.fit(model, data)
    trainer.test()
