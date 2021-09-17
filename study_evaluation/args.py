import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="",
        help="Slice level prediction file path",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="Evaluation mode (train/valid/test"
    )
    args = parser.parse_args()

    return args
