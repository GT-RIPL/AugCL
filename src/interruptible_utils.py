import os
import shutil
import os.path as osp
import signal
import threading
from algorithms.sac import SAC
from utils import ReplayBuffer

import torch

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", 0)
STATE_FOLDER = osp.join(os.environ["HOME"], ".interrupted_states")
BUFFER_FOLDER = osp.join(STATE_FOLDER, str(SLURM_JOB_ID))
STATE_FILE = osp.join(STATE_FOLDER, "{}.tar".format(SLURM_JOB_ID))

REQUEUE = threading.Event()
REQUEUE.clear()
EXIT = threading.Event()
EXIT.clear()


def _requeue_handler(signum, frame):
    # define the handler function
    # note that this is not executed here, but rather
    # when the associated signal is sent
    print("signaled for requeue")
    EXIT.set()
    REQUEUE.set()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def init_handlers():
    signal.signal(signal.SIGUSR1, _requeue_handler)

    signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)
    signal.signal(signal.SIGUSR2, _clean_exit_handler)


def save_state(agent: SAC, replay_buffer: ReplayBuffer, step: int):
    torch.save({"step": step, "agent": agent}, STATE_FILE)
    os.makedirs(BUFFER_FOLDER, exist_ok=True)
    replay_buffer.requeue_save(BUFFER_FOLDER)


def save_and_requeue(agent: SAC, replay_buffer: ReplayBuffer, step: int):
    save_state(agent=agent, replay_buffer=replay_buffer, step=step)
    print("requeuing job " + os.environ["SLURM_JOB_ID"])
    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])


def is_requeued():
    return osp.exists(STATE_FILE)


def requeue_load_agent_and_replay_buffer(replay_buffer: ReplayBuffer):
    state_dict = torch.load(STATE_FILE)
    replay_buffer.load(BUFFER_FOLDER)
    return state_dict["agent"], state_dict["step"]


def delete_requeue_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    if os.path.exists(BUFFER_FOLDER):
        shutil.rmtree(BUFFER_FOLDER)
