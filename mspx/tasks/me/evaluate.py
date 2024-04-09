#

import torch
import numpy as np
from typing import List
from collections import defaultdict, Counter

from transformers import AutoTokenizer
from easyeditor.util import HyperParams
from easyeditor.models.melo.melo import LORA as MELO_LORA
from .utils import get_first_param, get_sub_tokens
from .inst import Instance

# --
# perform evaluation

def evaluate_edit_quality(model, model_name: str, hparams: HyperParams, tok: AutoTokenizer, instance: Instance, eval_methods, eval_pre=None, prompt_prefix=None):
    if isinstance(model, MELO_LORA):
        model = model.model
    # --
    model.eval()
    ret = {}
    for key0 in ["edit", "rephrase", "portability", "locality"]:
        items = instance[key0]
        if len(items) == 0:
            continue  # no item to eval!
        prompts = [z.prompt for z in items]
        if prompt_prefix is not None:
            prompts = [f"{prompt_prefix}{p}" for p in prompts]
        this_eval_pre = None
        if key0 == "locality" and eval_pre is not None:  # slightly strange here for the locality checking, but ...
            this_eval_pre = []
            _counts = Counter()
            for one_idx, one_item in enumerate(items):
                kk = f"{key0}.{one_item.type if one_item.type else '__'}"
                this_eval_pre.append(eval_pre[kk][_counts[kk]])
                _counts[kk] += 1
        results = test_prediction_acc(model, tok, hparams, prompts, [z.answer for z in items], [z.answer_alias for z in items], eval_methods, eval_pre=this_eval_pre)
        assert len(results) == len(items)
        for one_idx, one_item in enumerate(items):
            kk = f"{key0}.{one_item.type if one_item.type else '__'}"
            if kk not in ret:
                ret[kk] = []
            ret[kk].append(results[one_idx])
    # --
    return ret

def test_prediction_acc(model, tok, hparams, prompts: List[str], targets: List[str], target_aliases: List[List[str]], eval_methods, eval_pre=None):
    # -- eval batch size (simply fix it!)
    _FORCE_BATCH = 4
    # --
    with torch.no_grad():
        _orig_padding_size = tok.padding_side
        tok.padding_side = "right"  # right padding for this eval!
        device = get_first_param(model).device
        ret = [{} for _ in range(len(prompts))]
        if "force" in eval_methods:  # teacher forcing: only care about targets!
            for base_pidx in range(0, len(prompts), _FORCE_BATCH):
                _i0, _i1 = base_pidx, min(len(prompts), base_pidx+_FORCE_BATCH)  # current batch
                _prompts, _targets = prompts[_i0:_i1], targets[_i0:_i1]
                orig_inputs = [prompt.strip() for prompt in _prompts]  # prefixes
                extended_inputs = [prompt.strip() + ' ' + target.strip() for prompt, target in zip(_prompts, _targets)]  # full inputs
                t_orig = tok(orig_inputs, padding=True, return_tensors="pt").to(device)
                t_ext = tok(extended_inputs, padding=True, return_tensors="pt").to(device)
                t_len_orig, t_len_ext = (t_orig['input_ids'] != tok.pad_token_id).sum(-1), (t_ext['input_ids'] != tok.pad_token_id).sum(-1)  # [bs]
                outputs = model(**t_ext)  # forward them all
                t_logits = outputs if type(outputs) is torch.Tensor else outputs.logits  # [bs, L_ext, V]
                t_argmax = t_logits.argmax(dim=-1)  # [bs, L]
                # we need to explicitly store the predictions for locality calculation!
                for pidx in range(_i0, _i1):
                    _pidx0 = pidx - _i0
                    t2_len0, t2_len1 = t_len_orig[_pidx0], t_len_ext[_pidx0]  # []
                    t2_gold = t_ext['input_ids'][_pidx0, t2_len0:t2_len1].tolist() if eval_pre is None else eval_pre[pidx]["force_argmax"]  # [Ls]
                    t2_pred = t_argmax[_pidx0, t2_len0-1:t2_len1-1].tolist()  # [Ls]
                    acc = np.mean([a==b for a,b in zip(t2_gold, t2_pred)]).item() if len(t2_gold) > 0 else 0.
                    ret[pidx].update({"force_acc": acc, "force_argmax": t2_pred, "force_gold": t2_gold})
        if "gen" in eval_methods:  # free generation
            for pidx, prompt in enumerate(prompts):  # todo(note): for simplicity no batching!
                if eval_pre is not None:
                    _prev_str = eval_pre[pidx]["gen_str"].split("\n")[0].strip()  # in case generating "\n"
                    potential_answers = [_prev_str]
                else:
                    potential_answers = [z.strip() for z in [targets[pidx]] + target_aliases[pidx]]
                max_new_tokens = max(len(get_sub_tokens(tok, a)) for a in potential_answers) + 2  # max tokens to generate (generate some more...)
                prompt_tok = tok(prompt, return_tensors="pt").to(device)
                gen_res = model.generate(**prompt_tok, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)[0].tolist()  # simply greedy
                _new_tokens = gen_res[prompt_tok["input_ids"].shape[-1]:]
                gen_str = tok.decode(_new_tokens).strip()  # only the output str
                # todo(note): not accurate with only prefix checking, but ...
                gen_score = float(any(gen_str.startswith(z) for z in potential_answers))
                ret[pidx].update({"gen_str": gen_str, "gen_tokens": _new_tokens, "gen_score": gen_score, "gen_golds": potential_answers})
        # breakpoint()
        tok.padding_side = _orig_padding_size  # restore padding
        return ret

# --
# summary evaluation

def get_res(vs):
    res = f"{sum(vs):.2f}/{len(vs)}={np.mean(vs).item():.4f}" if vs else "NIL"
    return res

def summary_metrics(metrics):
    # collect all scores: gen_score="greedy decoding + matching", force_accP="teacher forcing partial matching", force_accB="teacher forcing full matching"
    ets = ["force_accP", "force_accB", "gen_score"]  # eval types
    res_lists = {et: defaultdict(list) for et in ets}
    res_force_partial, res_force_binary, res_gen_score = [res_lists[z] for z in ets]
    for m in metrics:
        for kk, vvs in m.items():
            kk0, _ = kk.split(".")  # leval0
            for vv in vvs:
                if "force_acc" in vv:
                    _accP = vv["force_acc"]
                    _accB = float(_accP >= 1.)
                    res_force_partial[kk0].append(_accP)
                    res_force_partial[kk].append(_accP)
                    res_force_binary[kk0].append(_accB)
                    res_force_binary[kk].append(_accB)
                if "gen_score" in vv:
                    res_gen_score[kk0].append(vv["gen_score"])
                    res_gen_score[kk].append(vv["gen_score"])
    # gather final results
    final_ret = {}
    for one_et, one_res in res_lists.items():
        final_ret[one_et] = {k: get_res(one_res[k]) for k in sorted(one_res.keys())}
    _zres_str = "E={}||{}||{};;R={}||{}||{};;P={}||{}||{};;L={}||{}||{}"
    _zres_key = ["edit", "rephrase", "portability", "locality"]
    final_ret["zres_full"] = _zres_str.format(*[final_ret[z2].get(z1, "NA") for z1 in _zres_key for z2 in ets])
    final_ret["zres"] = _zres_str.format(*[final_ret[z2].get(z1, "NA").split("=")[-1] for z1 in _zres_key for z2 in ets])
    return final_ret
