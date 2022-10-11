## Algorithms

This repository contains implementations of the following algorithms in a unified framework:

- [SVEA (Hansen et al., 2021)](https://arxiv.org/abs/2107.00644)
- [SODA (Hansen and Wang, 2021)](https://arxiv.org/abs/2011.13389)
- [PAD (Hansen et al., 2020)](https://arxiv.org/abs/2007.04309)
- [DrQ (Kostrikov et al., 2020)](https://arxiv.org/abs/2004.13649)
- [RAD (Laskin et al., 2020)](https://arxiv.org/abs/2004.14990)
- [CURL (Srinivas et al., 2020)](https://arxiv.org/abs/2004.04136)
- [SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)
- [Non-Naive RAD (Hansen et al., 2021)](https://arxiv.org/abs/2107.00644)
- [AugCL]

using standardized architectures and hyper-parameters, wherever applicable. If you want to add an algorithm, feel free to send a pull request.


## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:

```
conda env create -f setup/conda.yml
conda activate dmcgb
sh setup/install_envs.sh
```

You will also need to setup MuJoCo and DeepMindControl https://www.deepmind.com/open-source/deepmind-control-suite


## Datasets
Part of this repository relies on external datasets. SODA uses the [Places](http://places2.csail.mit.edu/download.html) dataset for data augmentation, which can be downloaded by running

```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

If [Places] is unavailible there's also [CoCo](https://cocodataset.org/), which requires setting up an account with them.

Distracting Control Suite uses the [DAVIS](https://davischallenge.org/davis2017/code.html) dataset for video backgrounds, which can be downloaded by running

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

You should familiarize yourself with their terms before downloading. After downloading and extracting the data, add your dataset directory to the `datasets` list in `setup/config.cfg`.

The `video_easy` environment was proposed in [PAD](https://github.com/nicklashansen/policy-adaptation-during-deployment), and the `video_hard` environment uses a subset of the [RealEstate10K](https://google.github.io/realestate10k/) dataset for background rendering. All test environments (including video files) are included in this repository, namely in the `src/env/` directory.


## Training & Evaluation

The `scripts` directory contains training and evaluation bash scripts for all the included algorithms. Alternatively, you can call the python scripts directly, e.g. for training call

```
python3 src/train.py \
  --algorithm <algorithm name> \
  --seed 0
```

You can see the parameter key for an algorithm under src/algorithms/factory.py.

To run AugCL we first suggest training an agent using the command

```
python3 src/train.py \
 -- id <id>
 --algorithm non_naive_rad
 --data_aug shift
 --train_steps 200k
 --save_buffer True
```

Then for the strong augmentation phase after the above process is completed run

```
python3 src/train.py \
 --algorithm augcl
 --data_aug splice
 --curriculum_step 200000
 --prev_id <id>
 --prev_algorithm non_naive_rad
```

The above will search `logs` folder for weights and stored replay buffer of a model matching the `prev_id` and `prev_algorithm` with matching seed. Seed can be set with `--seed`.

To evaluate run `eval.py` with the same arguments as `train.py`, supported arguments can be see in `src/arguments.py`