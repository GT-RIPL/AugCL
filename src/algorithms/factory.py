from algorithms.darq import DArQ
from algorithms.darq_head import DArQ_head
from algorithms.darq_middle import DArQMiddle
from algorithms.drq2 import DrQ2
from algorithms.drq_mix import DrQMix
from algorithms.drq_rad import DrQRAD
from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.transfer import Transfer
from algorithms.transfer_full import TransferFull
from algorithms.wsa import WSA
from algorithms.sac_no_share import SAC_no_share
from algorithms.blah import BLAH
from algorithms.ftl import FTL
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
    "transfer": Transfer,
    "drq2": DrQ2,
    "drq_mix": DrQMix,
    "darq_middle": DArQMiddle,
    "darq_head": DArQ_head,
    "transfer_full": TransferFull,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
