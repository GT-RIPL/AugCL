import augmentations
from algorithms.sac import SAC


class CurriculumBackbone(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_func = augmentations.aug_to_func[args.data_aug]
