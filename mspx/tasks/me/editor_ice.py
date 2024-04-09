#

# helper for ICE (and related context-based methods)

from typing import List, Dict
from dataclasses import dataclass
import yaml

from easyeditor.util.hparams import HyperParams
from .inst import Item
from .utils import get_logger

@dataclass
class ICEHyperParams(HyperParams):
    # method specific
    template: str

    # Module templates
    device: int
    alg_name: str
    model_name: str
    model_parallel: bool = False
    fp16: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        assert (config and config['alg_name'] == 'ICE') or \
               print(f'KNHyperParams can not load from {hparams_name_or_path}, alg_name is {config["alg_name"]} ')
        return cls(**config)

LOG = get_logger()

class ICEHelper:
    def __init__(self, hparams: ICEHyperParams, model):
        self.hparams = hparams

    def apply_algo(self, model, tok, requests: List[Dict], hparams: ICEHyperParams):
        _template = hparams.template
        formatted_strs = [Item.format_dict(r) for r in requests]
        if _template == "imagine":
            _prefix = "".join(["Imagine that:\n"] + [(z+"\n") for z in formatted_strs] + ["Then:\n"])
        elif _template == "given":
            _prefix = "".join(["Given the following new facts:\n"] + [(z+"\n") for z in formatted_strs] + ["Then:\n"])
        else:
            raise NotImplementedError(f"UNK template: {_template}")
        return _prefix, model, {}
