import os
import pandas as pd
import argparse


def main(args):
    compiled_df = None
    for dir_path, method_name in zip(args.alg_dirs, args.method_names):
        for seed in range(args.max_seed + 1):
            alg_seed_path = os.path.join(dir_path, f"seed_{seed}")
            csv_path = os.path.join(alg_seed_path, args.csv_file_name)
            curr_df = pd.read_csv(csv_path)
            subset_df = curr_df[["step", args.metric]]
            subset_df.rename(columns={args.metric: "perf"})
            subset_df["seed"] = seed
            subset_df["method_name"] = method_name

            if compiled_df is None:
                compiled_df = subset_df
            else:
                compiled_df = compiled_df.append(subset_df)

    if not os.path.exists("./compiled_CSV"):
        os.makedirs("./compiled_CSV", exist_ok=True)
    compiled_df.to_csv(f"./compiled_CSV/{args.csv_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_name", default="No_name", type=str)
    parser.add_argument("--alg_dirs", "--list", nargs="+", required=True)
    parser.add_argument("--method_names", type=str, required=True)
    parser.add_argument("--max_seed", default=1, type=int)
    parser.add_argument("--metric", default="episode_reward", type=str)
    parser.add_argument("--csv_file_name", default="train.csv", type=str)
    args = parser.parse_args()
    args.method_names = args.method_names.split("-")
    main(args)
