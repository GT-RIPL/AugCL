from algorithms.darq import DArQ
from algorithms.drq_rad import DrQRAD
from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.wsa import WSA
from algorithms.sac_no_share import SAC_no_share
from algorithms.blah import BLAH
from algorithms.FTL import FTL
from algorithms.drq_rad import DrQRAD
from algorithms.darq import DArQ

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "wsa": WSA,
    "sac_no_share": SAC_no_share,
    "blah": BLAH,
    "ftl": FTL,
    "drqrad": DrQRAD,
    "darq": DArQ,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
