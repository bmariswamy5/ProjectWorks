import argparse
from datasets import load_dataset
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Download a percentage of the legal_contracts dataset.")
    parser.add_argument("--percent", type=int, default=10, help="Percentage of data to download (between 1 and 100)")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.percent < 1 or args.percent > 100:
        raise ValueError("Percentage must be between 1 and 100")

    split = f"train[:{args.percent}%]"

    dataset = load_dataset("albertvillanova/legal_contracts", split=split)

    save_path = os.path.join(os.getcwd(), "data")

    os.makedirs(save_path, exist_ok=True)

    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    main()
