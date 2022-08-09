from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.transfer import Transfer
from algorithms.drq_aug import DrQ_Aug
from algorithms.curriculum_learning.curriculum import Curriculum
from algorithms.non_naive_rad import NonNaiveRAD
from algorithms.curriculum_learning.WSDE import WSDE

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "transfer": Transfer,
    "drq_aug": DrQ_Aug,
    "curriculum": Curriculum,
    "non_naive_rad": NonNaiveRAD,
    "WSDE": WSDE,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
