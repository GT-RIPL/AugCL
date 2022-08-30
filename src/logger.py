from collections import defaultdict
import json
import shutil
import os
import torch
from termcolor import colored
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

test_env_reward_key = "episode_reward_"
FORMAT_CONFIG = {
    "rl": {
        "train": [
            ("episode", "E", "int"),
            ("step", "S", "int"),
            ("duration", "D", "time"),
            ("episode_reward", "R", "float"),
            ("actor_loss", "ALOSS", "float"),
            ("critic_loss", "CLOSS", "float"),
            ("aux_loss", "AUXLOSS", "float"),
        ],
        "eval": [
            ("step", "S", "int"),
            ("episode_reward", "ER", "float"),
            (test_env_reward_key, "ERTEST", "float"),
        ],
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating, continue_train=False):
        self._file_name = file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self.dict_list = list()
        self.continue_train = continue_train
        self.is_first_log = True

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, "a") as f:
            f.write(json.dumps(data) + "\n")
        self.dict_list.append(data)
        csv_fp = self._file_name[:-4] + ".csv"

        df = pd.DataFrame(self.dict_list)
        if os.path.exists(csv_fp) and self.continue_train and self.is_first_log:
            df_og = pd.read_csv(csv_fp)
            df = df_og.append(df)
            self.is_first_log = False

        df.to_csv(csv_fp, index=False)
        self.dict_list = list()

    def _format(self, key, value, ty):
        template = "%s: "
        if ty == "int":
            template += "%d"
        elif ty == "float":
            template += "%.04f"
        elif ty == "time":
            template += "%.01f s"
        else:
            raise "invalid format type: %s" % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = ["{:5}".format(prefix)]
        for key, disp_key, ty in self._formating:
            if key == test_env_reward_key:
                for key_data in data.keys():
                    if key_data.startswith(key):
                        key = key_data
                        break

            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print("| %s" % (" | ".join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["step"] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb=True, config="rl", continue_train=False):
        self._log_dir = log_dir
        if use_tb:
            tb_dir = os.path.join(log_dir, "tb")
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(log_dir, "train.log"),
            formating=FORMAT_CONFIG[config]["train"],
            continue_train=continue_train,
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, "eval.log"),
            formating=FORMAT_CONFIG[config]["eval"],
            continue_train=continue_train,
        )

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step, n=1):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value, n)

    def dump(self, step):
        self._train_mg.dump(step, "train")
        self._eval_mg.dump(step, "eval")
