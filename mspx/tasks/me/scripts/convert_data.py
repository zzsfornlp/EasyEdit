#

# convert various ME data to similar formats

import sys
import json
from collections import Counter

try:
    from mspx.utils import default_json_helper, Conf, init_everything, zlog, zwarn, ZHelper
except:
    print("Please use the mspx.utils here: https://github.com/zzsfornlp/zmsp/tree/main/mspx/utils")

class MainConf(Conf):
    def __init__(self):
        # for data convert
        self.input_format = ""  # input format
        self.input_file = ""
        self.output_file = ""
        # for data filtering
        self.filter_exclude_file = ""  # items to exclude

def do_filter(conf):
    cc = Counter()
    # --
    # sig_f = lambda x: tuple(sorted([tuple([z[k] for k in ["subject", "question", "answer", "answer_old"]]) for z in x['edit']]))
    # note: targets may be different!
    # sig_f = lambda x: tuple(sorted([tuple([z[k] for k in ["subject", "question", "answer_old"]]) for z in x['edit']]))
    sig_f = lambda x: x['info']['sig']
    exclude_sigs = set()
    for one_exclude in default_json_helper.from_auto(conf.filter_exclude_file):
        sig = sig_f(one_exclude)
        if sig in exclude_sigs:
            zwarn(f"Repeat sig: {sig}")
        exclude_sigs.add(sig)
        cc['exclude_read'] += 1
    cc['exclude_set'] = len(exclude_sigs)
    # --
    all_outputs = []
    for one_input in default_json_helper.from_auto(conf.input_file):
        cc['inst_input'] += 1
        sig1 = sig_f(one_input)
        if sig1 in exclude_sigs:
            cc['inst_exclude'] += 1
        else:
            all_outputs.append(one_input)
            cc['inst_include'] += 1
    default_json_helper.to_auto(conf.output_file, all_outputs)
    zlog(f"Filter: {conf.input_file} - {conf.filter_exclude_file} = {conf.output_file}: {cc}")

def do_convert(conf):
    cc = Counter()
    all_outputs = []
    for one_input in default_json_helper.from_auto(conf.input_file):
        cc['inst'] += 1
        # one_output = {}
        # --
        # zsre from editing-data.zip
        if conf.input_format.startswith("zsre1"):  # zsre from editing-data.zip
            # assert len(one_input["answers"]) == 1
            _edit = {
                "subject": one_input["subject"],
                "question": one_input["src"],
                "answer": one_input["alt"],
                "answer_old": one_input["answers"][0],
            }
            _rephrase = _edit.copy()
            _rephrase["question"] = one_input["rephrase"]
            one_output = {"edit": [_edit], "rephrase": [_rephrase]}
            if conf.input_format == "zsre1":  # original one
                _loc, _loc_ans = one_input["loc"], one_input["loc_ans"]
                assert _loc.startswith("nq question:")
                _loc = _loc[len("nq question:"):].strip()
                _loc_ans = _loc_ans.strip()
                if not _loc.endswith("?"):
                    _loc = _loc + "?"
                if str.islower(_loc[0]):
                    _loc = f"{str.upper(_loc[0])}{_loc[1:]}"
                if str.islower(_loc_ans[0]):
                    _loc_ans = f"{str.upper(_loc_ans[0])}{_loc_ans[1:]}"
                _locality = {
                    "type": "nq",
                    "question": _loc,
                    "answer": _loc_ans,
                }
                one_output["locality"] = [_locality]
            elif conf.input_format == "zsre1_invrel":  # Inverse relation
                _portability = {
                    "type": "inv_rel",
                    "question": one_input["inverse question"],
                    "answer": one_input["subject"],
                }
                one_output["portability"] = [_portability]
            elif conf.input_format == "zsre1_onehop":  # One hop
                _portability = {
                    "type": "onehop",
                    "question": one_input["portability"]["New Question"],
                    "answer": one_input["portability"]["New Answer"],
                }
                one_output["portability"] = [_portability]
            elif conf.input_format == "zsre1_altsubj":  # Alt subject
                _portability = {
                    "type": "altsubj",
                    "subject": one_input["alternative_subject"],
                    "question": one_input["alter_subject_question"],
                    "answer": one_input["alt"],
                }
                one_output["portability"] = [_portability]
            else:
                raise RuntimeError(f"UNK format of {conf.input_format}")
        # --
        # counterfact from editing-data.zip
        elif conf.input_format == "cf1":  # counterfact from editing-data.zip
            _edit = {
                "subject": one_input["subject"],
                "question": one_input["prompt"],
                "answer": one_input["target_new"],
                "answer_old": one_input["ground_truth"],
            }
            _rephrase = _edit.copy()
            _rephrase["question"] = one_input["rephrase_prompt"]
            _locality = {
                "type": "default",
                "question": one_input["locality_prompt"],
                "answer": one_input["locality_ground_truth"],
            }
            one_output = {"edit": [_edit], "rephrase": [_rephrase], "locality": [_locality]}
        # --
        # KnowEdit1: with zsre, wiki_recent, wiki_counterfact
        elif conf.input_format == "knowedit1":
            _edit = {
                "subject": one_input["subject"],
                "question": one_input["prompt"],
                "answer": one_input["target_new"],
                # "answer_old": one_input["ground_truth"],
            }
            if "groud_truth" in one_input:
                _gt = one_input["ground_truth"]
                if isinstance(_gt, list):
                    _gt = _gt[0]
                _edit["answer_old"] = _gt
            one_output = {"edit": [_edit], "rephrase": [], "portability": [], "locality": []}
            _rephrase_prompts = [one_input[z] for z in ["rephrase", "rephrase_prompt"] if z in one_input]
            assert len(_rephrase_prompts) <= 1
            if len(_rephrase_prompts) == 1:
                _rephrase = _edit.copy()
                _rephrase["question"] = _rephrase_prompts[0]
                one_output["rephrase"].append(_rephrase)
            for key0 in ["portability", "locality"]:
                if key0 not in one_input:
                    continue
                for key1 in one_input[key0].keys():  # type of testing
                    for one_input_case in one_input[key0][key1]:
                        _all_answers = []
                        _gts = one_input_case["ground_truth"]
                        if isinstance(_gts, str):
                            _gts = [_gts]
                        for _one_answers in _gts:
                            if isinstance(_one_answers, str):
                                _all_answers.append(_one_answers)
                            else:
                                assert isinstance(_one_answers, list)
                                _all_answers.extend(_one_answers)
                        _one_item = {
                            "type": key1,
                            "question": one_input_case["prompt"],
                            "answer": _all_answers[0],  # note: simply put the first one
                            # "answer_alias": _all_answers[1:],
                        }
                        if len(_all_answers) > 1:
                            _one_item["answer_alias"] = _all_answers[1:]
                        one_output[key0].append(_one_item)
        # --
        # MQUAKE & RippleEdits
        elif conf.input_format == "MQUAKE":
            cc[f'inst_MQUAKE_edit={len(one_input["requested_rewrite"])}'] += 1
            cc[f'inst_MQUAKE_hop={len(one_input["single_hops"])}'] += 1  # some stat
            # --
            one_output = {"edit": [], "rephrase": [], "portability": [], "locality": []}
            for one_edit in one_input["requested_rewrite"]:
                _edit = {
                    "subject": one_edit["subject"],
                    "question": one_edit["question"],  # still use the question
                    "answer": one_edit["target_new"]["str"],
                    "answer_old": one_edit["target_true"]["str"],
                }
                one_output["edit"].append(_edit)
            num_hop = len(one_input["single_hops"])
            for one_question in one_input["questions"]:
                _portability = {
                    "type": f"hop{num_hop}",
                    "question": one_question,
                    "answer": one_input["new_answer"],
                    "answer_alias": one_input["new_answer_alias"],
                    "answer_old": one_input["answer"],
                    "answer_old_alias": one_input["answer_alias"],
                }
                one_output["portability"].append(_portability)
            one_output["info"] = {"sig": "_".join([f"{one_edit['target_true']['id']}+{one_edit['relation_id']}" for one_edit in one_input["requested_rewrite"]])}
        elif conf.input_format == "RippleEdits":
            assert False
            # breakpoint()  # TODO(!): simply use this!
            pass
        # --
        else:
            raise RuntimeError(f"UNK format of {conf.input_format}")
        # --
        # clear spaces
        one_output = ZHelper.recursive_apply(one_output, f_target=(lambda x: isinstance(x, str)), f_modify=(lambda x: x.strip()))
        if "info" not in one_output:
            one_output["info"] = {}
        one_output["info"].update({"_orig": json.dumps(one_input)})
        # --
        # further count
        for key0 in ["edit", "rephrase", "portability", "locality"]:
            for one_item in one_output.get(key0, []):
                cc[f"item_{key0}_{one_item.get('type', '')}"] += 1
        # --
        if any((z['subject'] not in z['question']) for z in one_output["edit"]):
            zwarn(f"Subject not in prompt: {one_output}")
            cc['inst_bad'] += 1
        else:
            all_outputs.append(one_output)
        # --
    default_json_helper.to_auto(conf.output_file, all_outputs)
    zlog(f"Read from {conf.input_file} to {conf.output_file}: {cc}")

def main():
    conf: MainConf = init_everything(MainConf())
    if conf.filter_exclude_file:
        do_filter(conf)
    else:
        do_convert(conf)
    # --

# python -m mspx.tasks.me.scripts.convert_data [format] IN OUT
if __name__ == '__main__':
    main()

"""
{
# --
#@dataME -- batch1 (from model-editing.zip)
python -m mspx.tasks.me.scripts.convert_data input_format:zsre1 input_file:data_editing/zsre/zsre_mend_eval.json output_file:zsre1.eval.json
python -m mspx.tasks.me.scripts.convert_data input_format:zsre1 input_file:data_editing/zsre/zsre_mend_train_10000.json output_file:zsre1.train_10000.json
python -m mspx.tasks.me.scripts.convert_data input_format:cf1 input_file:data_editing/counterfact/counterfact-train.json output_file:cf1.train.json
python -m mspx.tasks.me.scripts.convert_data input_format:cf1 input_file:data_editing/counterfact/counterfact-edit.json output_file:cf1.eval.json
python -m mspx.tasks.me.scripts.convert_data input_format:zsre1_invrel --input_file data_editing/portability/Inv*/zsre* -- output_file:zsre1_invrel.eval.json
python -m mspx.tasks.me.scripts.convert_data input_format:zsre1_onehop --input_file data_editing/portability/One*/zsre* -- output_file:zsre1_onehop.eval.json
python -m mspx.tasks.me.scripts.convert_data input_format:zsre1_altsubj --input_file data_editing/portability/Sub*/zsre* -- output_file:zsre1_altsubj.eval.json
# zsre1.eval=19086, zsre1.train_10000=10000, cf1.train=cf1.eval=10000, zsre1_invrel=385(-5), zsre1_onehop=1037(-0), zsre1_altsubj=293(-4)
# --
#@dataME -- batch1.5 (from KnowEdit1)
for ff in KnowEdit/benchmark/{wiki*,ZsRE}/*.json; do
python -m mspx.tasks.me.scripts.convert_data input_format:knowedit1 input_file:$ff "output_file:knowedit1.`basename $ff`"
done
# --
#@dataME -- batch2 (from MQUAKE & RippleEdits)
for ff in MQuAKE/datasets/*.json; do
python -m mspx.tasks.me.scripts.convert_data input_format:MQUAKE input_file:$ff output_file:`basename $ff`
done
# CF excluding 3K
python -m mspx.tasks.me.scripts.convert_data input_format:MQUAKE input_file:MQuAKE-CF.json filter_exclude_file:MQuAKE-CF-3k.json output_file:MQuAKE-CF-5k.json
} |& tee _log_data0
"""
