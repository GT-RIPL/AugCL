import augmentations
from algorithms.sac import SAC

class TransferBackbone(SAC):  # [K=1, M=1]
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_func = augmentations.aug_to_func[args.transfer_data_augs]
