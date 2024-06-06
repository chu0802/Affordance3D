import json
import os
import random

import cv2
import numpy as np
import omegaconf
import torch
import torch.backends.cudnn


def omegaconf_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if v is None:
                res[k] = v
            elif isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
            elif isinstance(v, omegaconf.ListConfig):
                res[k] = omegaconf.OmegaConf.to_container(v, resolve=True)
            else:
                raise RuntimeError(
                    "[!] The type of {} is not supported.".format(type(v))
                )
        return res

    return _to_dot_dict(hparams)


def load_view_points(scene_name, prompt, filename="view_points.json"):
    with open(filename) as f:
        data = json.load(f)[scene_name][prompt]

    return data["view_points"], data["look_at"]


def fix_seed(seed=0):
    cuda_device_id = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
