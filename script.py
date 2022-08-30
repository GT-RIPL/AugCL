import os
import argparse


def create_action_repeat_params(domain: str):
    if domain == "cartpole":
        return "--action_repeat 8 "
    elif domain == "finger":
        return "--action_repeat 2 "
    else:
        return ""


def main(args):
    seeds = range(args.max_seed + 1)
    domain_tasks = []

    for splits in args.tasks.split(","):
        domain_tasks.append(tuple(splits.split("-")))

    for domain, task in domain_tasks:
        task_acronym = f"{domain[0]}{task[0]}_"
        DT_full_name = f"{domain}_{task}"
        domain_task_params = f"--domain_name {domain} --task_name {task} "
        name = task_acronym + args.bash_file_sub + "_seed{}"
        action_repeat_param = create_action_repeat_params(domain=domain)

        for seed in seeds:
            curr_name = name.format(str(seed))
            fp = f"{DT_full_name}/{curr_name}.sh"
            with open(fp, "w") as f:
                bash_file_str = "#!/bin/bash -l\n#SBATCH --gres=gpu:1\n"
                if args.use_long:
                    bash_file_str += "#SBATCH -p long\n"
                else:
                    bash_file_str += "#SBATCH -p overcap\n#SBATCH -A overcap\n#SBATCH --signal=USR1@1800\n#SBATCH --requeue\n"

                bash_file_str += f"#SBATCH -o err_{curr_name}.out\ncd /nethome/dyung6\nsource .dmcbash\nconda activate dmcgb_og\nexport MUJOCO_GL = 'egl'\ncd /nethome/dyung6/share4_dyung6/dmcontrol-generalization-benchmark\n"
                bash_file_str += (
                    f"srun python src/train.py --seed {seed} "
                    + domain_task_params
                    + action_repeat_param
                    + args.python_args
                )
                f.write(bash_file_str)

            if args.run_scripts:
                os.system(f"sbatch {fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_long",
        default=False,
        type=bool,
        help="If not set to overcap will set to long",
    )
    parser.add_argument("--bash_file_sub", required=True, type=str)
    parser.add_argument("--max_seed", default=5, type=int)
    parser.add_argument("--python_args", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Should be in <domain-task> format. For multiple tasks delimit with ,",
    )
    parser.add_argument("--run_scripts", default=False, type=bool)
    args = parser.parse_args()
    main(args)
