import pytorch_lightning as pl

from ..utils import object_from_dict


class SegmentationModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        if self.hparams.model["activation"] is None:
            raise ValueError("Please specify an activation.")

        if self.hparams.loss["from_logits"]:
            raise ValueError("Please set from_logits=False.")

        self.model = object_from_dict(self.hparams.model)
        self.loss = object_from_dict(self.hparams.loss)
        for metric in self.hparams.metrics:
            metric_name = metric["type"].split(".")[-1].lower()
            setattr(self, f"train_{metric_name}", object_from_dict(metric))
            setattr(self, f"val_{metric_name}", object_from_dict(metric))
            setattr(self, f"test_{metric_name}", object_from_dict(metric))

    def forward(self, x):
        return self.model(x)

    # def _step(self, batch, batch_idx):
    #     x, y = batch
    #     preds = self(x)
    #     loss = self.loss(preds, y)
    #     return {"loss": loss, "preds": preds, "target": y}

    # def _step_end(self, outputs, name):
    #     self.log(f"{name}_loss", outputs["loss"])
    #     for metric in self.hparams.metrics:
    #         metric_name = metric["type"].split(".")[-1].lower()
    #         value = getattr(self, f"{name}_{metric_name}")(
    #             outputs["preds"], outputs["target"]
    #         )
    #         self.log(f"{name}_{metric_name}", value)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        return {"loss": loss, "preds": preds, "target": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        return {"loss": loss, "preds": preds, "target": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):
        name = "train"
        self.log(f"{name}_loss", outputs["loss"])
        for metric in self.hparams.metrics:
            metric_name = metric["type"].split(".")[-1].lower()
            value = getattr(self, f"{name}_{metric_name}")(
                outputs["preds"], outputs["target"]
            )
            self.log(f"{name}_{metric_name}", value)

    def validation_step_end(self, outputs):
        name = "val"
        self.log(f"{name}_loss", outputs["loss"])
        for metric in self.hparams.metrics:
            metric_name = metric["type"].split(".")[-1].lower()
            value = getattr(self, f"{name}_{metric_name}")(
                outputs["preds"], outputs["target"]
            )
            self.log(f"{name}_{metric_name}", value)

    def test_step_end(self, outputs):
        name = "test"
        self.log(f"{name}_loss", outputs["loss"])
        for metric in self.hparams.metrics:
            metric_name = metric["type"].split(".")[-1].lower()
            value = getattr(self, f"{name}_{metric_name}")(
                outputs["preds"], outputs["target"]
            )
            self.log(f"{name}_{metric_name}", value)

    def configure_optimizers(self):
        optimizer = object_from_dict(self.hparams.optimizer, params=self.parameters())
        scheduler = object_from_dict(
            self.hparams.scheduler["scheduler"],
            optimizer=optimizer,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": self.hparams.scheduler["interval"],
            "frequency": self.hparams.scheduler["frequency"],
        }
        return [optimizer], [scheduler_dict]
