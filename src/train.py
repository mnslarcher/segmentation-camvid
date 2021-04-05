import argparse

import pytorch_lightning as pl
from albumentations.core.serialization import from_dict

from anomaly_segmentation.io import read_yaml
from anomaly_segmentation.lightning import SegmentationDataModule, SegmentationModule
from anomaly_segmentation.utils import object_from_dict


def _parse_args():
    parser = argparse.ArgumentParser("Train.")
    parser.add_argument(
        "--config-path", type=str, required=True, help="path to the configuration file."
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = _parse_args()
    hparams = read_yaml(args.config_path)
    pl.seed_everything(hparams["seed"])
    hparams["data"]["transforms"] = {
        "train": from_dict(hparams["transforms"]["train"]),
        "val": from_dict(hparams["transforms"]["val"]),
        "test": from_dict(hparams["transforms"]["test"]),
    }
    data = SegmentationDataModule(**hparams["data"])
    model = SegmentationModule(hparams["model"])
    trainer = object_from_dict(
        hparams["trainer"],
        logger=object_from_dict(hparams["logger"]),
        callbacks=[object_from_dict(callback) for callback in hparams["callbacks"]],
    )
    trainer.fit(model, data)
    trainer.test()
