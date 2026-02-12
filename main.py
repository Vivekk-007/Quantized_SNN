import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_snn", "train_ann", "eval"],
    )

    args = parser.parse_args()

    if args.mode == "train_snn":
        os.system("python -m train.train_snn")

    elif args.mode == "train_ann":
        os.system("python -m train.train_ann")

    elif args.mode == "eval":
        os.system("python -m eval.evaluate")


if __name__ == "__main__":
    main()

