import argparse
from typing import List, Tuple

from albumentations.core.serialization import from_dict

from anomaly_segmentation.io import read_yaml
from anomaly_segmentation.preprocessing import DataPreparer


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser("Prepare data.")
    parser.add_argument(
        "--config-path",
        type=str,
        metavar="N",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = _parse_args()
    kwargs = read_yaml(args.config_path)
    kwargs["transforms"] = from_dict(kwargs["transforms"])
    dp = DataPreparer(**kwargs)
    dp.prepare()
