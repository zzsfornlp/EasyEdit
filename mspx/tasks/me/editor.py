#

# simplify things from editor.py

import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer

from easyeditor.util.alg_dict import *
from easyeditor.util.hparams import HyperParams
from easyeditor.models.melo.melo import LORA as MELO_LORA
from .utils import get_logger, get_parameter
from .inst import Instance
from .evaluate import evaluate_edit_quality

LOG = get_logger()

class WrappedTokenizer:
    def __init__(self, tokenizer, add_special_tokens=True):
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def __call__(self, *args, **kwargs):
        add_special_tokens = kwargs.pop("add_special_tokens", self.add_special_tokens)
        return self.tokenizer.__call__(*args, add_special_tokens=add_special_tokens, **kwargs)

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)

    @property
    def padding_side(self):
        return self.tokenizer.padding_side

    @padding_side.setter
    def padding_side(self, x):
        self.tokenizer.padding_side = x

class MyEditor:
    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, print('Error: hparams is None.')
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT.get(hparams.alg_name, None)
        self.alg_name = hparams.alg_name
        LOG.info("Instantiating model")
        # get model and toker
        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                        device_map=device_map)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                              device_map=device_map)
                self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, fp32=False, trust_remote_code=True,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>',
                                                         pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                                                         trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError
            # --
            # note: specific setting of padding side: used in specific methods!
            if self.tok is not None and (
                    isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(
                    self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower()) and (
                    hparams.alg_name in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name
        # --
        if getattr(hparams, "no_add_special_tokens", False):
            self.tok = WrappedTokenizer(self.tok, add_special_tokens=False)
            LOG.info("Wrap tokenizer to get rid of special tokens")
        # --
        # move model to target
        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')
        self.hparams = hparams

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]

    def edit(self, requests, keep_original_weight: bool, verbose=True, edit_size=1, pre_file=None, eval_methods=("gen", "force"), train_insts=None, **kwargs):
        # process inputs
        if not isinstance(requests, list):
            requests = [requests]
        instances: List[Instance] = [(z if isinstance(z, Instance) else Instance.create(z)) for z in requests]  # make them objects
        if train_insts is not None:
            train_insts: List[Instance] = [(z if isinstance(z, Instance) else Instance.create(z)) for z in train_insts]  # make them objects
        # --
        # special modes
        ike_helper = None
        ice_helper = None
        if self.alg_name == "IKE":
            from .editor_ike import IKEHelper
            ike_helper = IKEHelper(self.hparams, self.model, train_insts)
        elif self.alg_name == "ICE":
            from .editor_ice import ICEHelper
            ice_helper = ICEHelper(self.hparams, self.model)
        # --
        # calculate pre results
        if pre_file and os.path.exists(pre_file):
            with open(pre_file) as fd:  # load the pre-calculated ones if we can
                all_metrics = json.load(fd)
        else:
            all_metrics = []
            idx_eval = 0
            for _eval_inst in tqdm(instances):  # calculate pre-calculated stuffs
                metrics = {
                    'case_id': idx_eval,
                    "requested_rewrite": _eval_inst.to_json(),
                    "pre": evaluate_edit_quality(self.model, self.model_name, self.hparams, self.tok, _eval_inst, eval_methods, prompt_prefix=""),
                }
                all_metrics.append(metrics)
                idx_eval += 1
            if pre_file:  # write the pre-calculated ones!
                with open(pre_file, 'w') as fd:
                    json.dump(all_metrics, fd, ensure_ascii=False, indent=2)
        assert len(all_metrics) == len(instances)
        # --
        idx_eval = 0
        for _edit_insts in self._chunks(instances, edit_size):
            # apply editing
            _exec_start = time()
            # convert to easy-editing format to use the algs!
            _converted_insts = sum((_inst.get_edit_insts() for _inst in _edit_insts), [])  # note: only need these fields for editing!
            _prompt_prefix = None
            if self.alg_name == "IKE":
                if len(_edit_insts) > 1:
                    LOG.warning("Currently this method (IKE) does not fit well with batch editing!")
                _prompt_prefix = ike_helper.retrieve_ike_facts(_converted_insts)
                edited_model, weights_copy = self.model, {}
            elif self.alg_name == "ICE":
                _prompt_prefix, edited_model, weights_copy = ice_helper.apply_algo(
                    self.model, self.tok, _converted_insts, self.hparams)
            else:
                self.model.train()
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    _converted_insts,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                )
            _exec_time = time() - _exec_start
            LOG.info(f"Execution editing: batch-size={len(_converted_insts)}, time={_exec_time}")
            # eval
            for _eval_inst in _edit_insts:
                assert idx_eval == all_metrics[idx_eval]["case_id"] and _eval_inst.to_json() == all_metrics[idx_eval]["requested_rewrite"]
                _eval_start = time()
                res_post = evaluate_edit_quality(self.model, self.model_name, self.hparams, self.tok, _eval_inst, eval_methods, eval_pre=all_metrics[idx_eval]["pre"], prompt_prefix=_prompt_prefix)
                _eval_time = time() - _eval_start
                all_metrics[idx_eval].update({"post": res_post, "time_exec": _exec_time, "time_eval": _eval_time})
                del all_metrics[idx_eval]["requested_rewrite"]
                if verbose:
                    LOG.info(f"Num #{idx_eval} editing: {all_metrics[idx_eval]}")
                idx_eval += 1
            # revert models back?
            if self.alg_name == 'KN' or (self.alg_name == 'GRACE' and keep_original_weight):
                with torch.no_grad():
                    weights_copy()  # unpatch_fn
            elif self.alg_name == 'LoRA' and keep_original_weight:
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            elif self.alg_name == 'LoRA' and not keep_original_weight:
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        p = get_parameter(self.model, k)
                        p[...] = v.to(p)
        # --
        if isinstance(edited_model, MELO_LORA):
            edited_model = edited_model.model
        return all_metrics, edited_model, weights_copy
