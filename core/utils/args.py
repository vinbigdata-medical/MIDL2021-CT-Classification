import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Config.yaml file path")
    parser.add_argument("--load", type=str, default="", help="model weight path")
    parser.add_argument(
        "--mode", type=str, default="train", help="model running mode (train/valid/test"
    )
    args = parser.parse_args()

    return args
