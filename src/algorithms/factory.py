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
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
