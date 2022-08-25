import os
import utils as utils
import numpy as np
from env.wrappers import make_env
from arguments import add_SAC_args, format_args


def main(args):
    if args.eval_mode is not None:
        env = make_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed + 42,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            image_size=400,
            mode=args.eval_mode,
            intensity=args.distracting_cs_intensity,
            frame_stack=1,
            background_on=not args.background_off,
            color_on=not args.color_off,
            camera_on=not args.camera_off,
        )
    else:
        raise ValueError(
            f"'eval_mode' in args is None, but mode needs to be set in order to sample observations"
        )

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.sample_frames_capacity,
        batch_size=args.num_samples,
    )

    done = True

    for _ in range(args.sample_frames_capacity):
        if done:
            obs = env.reset()
            done = False
        else:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

    idxs_samples = replay_buffer._get_idxs()
    obses, _ = replay_buffer._encode_obses(idxs=idxs_samples)

    dir = os.path.join(
        args.root_samples_dir,
        args.domain_name,
        args.task_name,
        args.eval_mode,
        str(args.distracting_cs_intensity) if args.distracting_cs_intensity else "",
    )
    os.makedirs(dir, exist_ok=True)

    for i in range(obses.shape[0]):
        img_pil = utils.numpy2pil(np.transpose(obses[i], (1, 2, 0)))
        img_pil.save(f"{dir}/sample_{i+1}.png")


if __name__ == "__main__":
    parser = add_SAC_args()
    parser.add_argument("--sample_frames_capacity", default=100, type=int)
    parser.add_argument("--num_samples", default=4, type=int)
    parser.add_argument("--root_samples_dir", default="./samples", type=str)
    args = parser.parse_args()
    args = format_args(args=args)
    main(args)
