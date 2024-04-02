#

# helpers

import logging
import os

import torch
import numpy as np
import random

# --
# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def get_logger():
    return LOGGER

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)

def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)

def get_first_param(model):
    p = next(model.parameters())
    return p

def get_sub_tokens(toker, s: str, convert_id=True):
    _toks = toker.tokenize(s)
    while len(_toks) > 0 and _toks[0] in ['▁', 'Ġ']:  # for llama and gpt2
        _toks = _toks[1:]
    # in some cases, there can be empty strings -> put the original word
    if len(_toks) == 0:
        _toks = [s]
    if convert_id:  # conver to ids
        ret = toker.convert_tokens_to_ids(_toks)
    else:
        ret = _toks
    return ret

def get_hparam_file(name):
    if os.path.exists(name):
        return name
    hparams_dir = os.path.join(os.path.dirname(__file__), "hparams")
    potential_filename = os.path.join(hparams_dir, name)
    if os.path.exists(potential_filename):
        return potential_filename
    else:
        raise FileNotFoundError(f"UNK filename: {name}")
