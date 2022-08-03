import os
import argparse
from utils import config_json_to_args
from algorithms.factory import algorithm


def add_SAC_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--domain_name", default="walker")
    parser.add_argument("--task_name", default="walk")
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--train_mode", default="train", type=str)
    parser.add_argument("--train_distracting_cs_intensity", default=0.0, type=float)

    # agent
    parser.add_argument("--algorithm", default="sac", type=str)
    parser.add_argument("--train_steps", default="500k", type=str)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_steps", default=1000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)

    # actor
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)

    # critic
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)

    # architecture
    parser.add_argument("--num_shared_layers", default=11, type=int)
    parser.add_argument("--num_head_layers", default=0, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--projection_dim", default=100, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)

    # entropy maximization
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument("--aux_lr", default=1e-3, type=float)
    parser.add_argument("--aux_beta", default=0.9, type=float)
    parser.add_argument("--aux_update_freq", default=2, type=int)

    # SODA
    parser.add_argument("--soda_batch_size", default=256, type=int)
    parser.add_argument("--soda_tau", default=0.005, type=float)

    # SVEA
    parser.add_argument("--svea_alpha", default=0.5, type=float)
    parser.add_argument("--svea_beta", default=0.5, type=float)

    # SVEA and SODA
    parser.add_argument("--use_overlay", default=False, type=bool)

    # DrQ and DrQ inspired
    parser.add_argument("--drq_k", default=1, type=int)
    parser.add_argument("--drq_m", default=1, type=int)

    # Data Augmentation
    parser.add_argument("--data_aug", default="identity", type=str)

    # eval
    parser.add_argument("--save_freq", default="100k", type=str)
    parser.add_argument("--eval_freq", default="10k", type=str)
    parser.add_argument("--eval_episodes", default=1, type=int)
    parser.add_argument("--eval_episodes_final_step", default=30, type=int)
    parser.add_argument("--eval_mode", default="color_hard", type=str)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)

    # misc
    parser.add_argument("--id", default="no_id", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--save_video", default=False, action="store_true")
    parser.add_argument("--save_buffer", default=False, type=bool)
    parser.add_argument("--continue_train", default=False, type=bool)
    parser.add_argument("--test_code_mode", default=False, type=bool)

    return parser


def assert_distracting_cs_intensity_valid(intensity: float):
    intensities = {0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    assert (
        intensity in intensities
    ), f"distracting_cs has only been implemented for intensities: {intensities}"


def assert_env_mode_valid(eval_mode: str):
    env_modes = {
        "train",
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
        "distracting_cs",
        "none",
    }

    assert (
        eval_mode in env_modes
    ), f'specified train mode "{eval_mode}" is not supported'


def format_args(args):
    if args.config_path:
        args = config_json_to_args(args=args, config_path=args.config_path)

    assert args.algorithm in frozenset(
        algorithm.keys()
    ), f'specified algorithm "{args.algorithm}" is not supported'

    assert_env_mode_valid(args.train_mode)
    assert_env_mode_valid(args.eval_mode)

    assert_distracting_cs_intensity_valid(intensity=args.distracting_cs_intensity)
    assert_distracting_cs_intensity_valid(intensity=args.train_distracting_cs_intensity)

    args.train_steps = int(args.train_steps.replace("k", "000"))
    args.save_freq = int(args.save_freq.replace("k", "000"))
    args.eval_freq = int(args.eval_freq.replace("k", "000"))

    if args.eval_mode == "none":
        args.eval_mode = None

    if args.algorithm in {"rad", "curl", "pad", "soda"}:
        args.image_size = 100
        args.image_crop_size = 84

        if args.algorithm == "rad" and "crop" not in args.data_aug:
            args.image_size = 84
    else:
        args.image_size = 84
        args.image_crop_size = 84

    return args


def parse_args():
    parser = add_SAC_args()
    args = parser.parse_args()
    return format_args(args=args)


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True)
    parser.add_argument("--env_mode", default="color_hard", type=str)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)
    parser.add_argument("--num_episodes", default=30, type=int)

    args = parser.parse_args()
    env_mode = args.env_mode
    distracting_cs_intensity = args.distracting_cs_intensity
    assert_distracting_cs_intensity_valid(intensity=distracting_cs_intensity)
    assert_env_mode_valid(eval_mode=env_mode)
    assert os.path.exists(
        os.path.join(args.dir_path, "args.json")
    ), f"{os.path.join(args.dir_path, 'args.json')} isn't a valid args.json path"

    return args


def parse_Q_variance_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--num_aug", default=10, type=int)

    args = parser.parse_args()
    assert os.path.exists(
        os.path.join(args.dir_path, "args.json")
    ), f"{os.path.join(args.dir_path, 'args.json')} isn't a valid args.json path"

    return args
