import json

import omegaconf


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
