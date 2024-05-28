import argparse
from datasets import load_dataset
from utils import update_args

def does_ds_exceed_threshold(args):
    ds = load_dataset(args.synth_ds_id, split=args.synth_ds_split)
    total_datapoints = len(ds)

    if total_datapoints > args.threshold:
        return "true"
    else:
        return "false"

def main():
    parser = argparse.ArgumentParser(description="Check Hugging Face dataset size.")
    parser.add_argument("--from-config", type=str, default="config/synth_data_gen.yaml",
                        help="set CLI options from YAML config")
    parser.add_argument("--synth-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID that synthetic dataset to be pushed")
    parser.add_argument("--synth-ds-split", type=str, default="eval",
                        help="Split of the synthetic dataset")
    parser.add_argument("--threshold", type=int, help="Threshold number of data points.")
    args = parser.parse_args()
    args = update_args(parser, args)

    print(does_ds_exceed_threshold(args))


if __name__ == "__main__":
    main()