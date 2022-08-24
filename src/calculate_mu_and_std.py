import os
import numpy as np
import pandas as pd
import argparse

from torch import int64


def main(args):
    values_list = []
    print(f"Generating Mu and Std of {args.column_name} over {args.max_seed}")
    for seed in range(args.max_seed + 1):
        csv_fp = os.path.join(args.dir_path, f"seed_{seed}", args.csv_file_name)
        df = pd.read_csv(csv_fp)

        if np.int64 == df[args.column_name].dtype:
            column_value = int(args.column_value)
        else:
            column_value = str(args.column_value)

        values_list.append(
            df.loc[df[args.column_name] == column_value][args.row_value].iloc[0]
        )

    print(f"Mean: {np.mean(values_list)}")
    print(f"Standard Deviation: {np.std(values_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True, type=str)
    parser.add_argument("--max_seed", default=1, type=int)
    parser.add_argument("--csv_file_name", default="eval.csv", type=str)
    parser.add_argument("--column_name", type=str, default="step")
    parser.add_argument("--column_value", default=500000)
    parser.add_argument("--row_value", required=True)
    args = parser.parse_args()
    main(args)
