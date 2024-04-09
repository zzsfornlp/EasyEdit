#

# run easy edit with unified data format
# note: with the easy-edit environment

import os
import os.path
import sys
import json
import torch
import pickle
import argparse
import logging

from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
)

from .editor import MyEditor, LOG
from .evaluate import summary_metrics
from .utils import get_hparam_file
from .editor_ice import ICEHyperParams
from .inst import Item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', default=None, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument("--path_test", required=True, type=str)
    parser.add_argument("--path_train", default=None, type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--prefile_dir', default=None, type=str)
    parser.add_argument('--edit_size', default=1, type=int)
    parser.add_argument('--sum_metric_files', default=None, type=str, nargs='+')
    parser.add_argument('--data_template', default="", type=str)  # template for the data
    parser.add_argument('--no_add_special_tokens', default=0, type=int)  # template for the data
    # other things to overwrite hparams
    parser.add_argument('--fp16', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    # --
    args = parser.parse_args()

    # --
    # summary mode! -- --editing_method 1 --hparams_dir 1 --path_test 1 --sum_metric_files ...
    if args.sum_metric_files:
        for ff in args.sum_metric_files:
            with open(ff) as fd:
                metrics = json.load(fd)
                summary = summary_metrics([z['post'] for z in metrics])
                LOG.info(f"Final result summary [L={len(metrics)}]: {ff}: {summary}")
        return
    # --

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'ICE':
        editing_hparams = ICEHyperParams
    else:
        raise NotImplementedError

    hparams = editing_hparams.from_hparams(get_hparam_file(args.hparams_dir))
    _suffix = f"{hparams.model_name.split('/')[-1]}_{os.path.basename(args.path_test).replace('.', '__')}"
    if args.ds_size:
        _suffix = f"{_suffix}_D{args.ds_size}"
    if args.data_template:
        _suffix = f"{_suffix}_T{args.data_template}"
        Item.set_prompt_template(args.data_template)
    if args.no_add_special_tokens:
        _suffix = f"{_suffix}_NST"
        hparams.no_add_special_tokens = True
    run_name = f"{args.editing_method}_{_suffix}"
    pre_file_name = _suffix

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # with open(os.path.join(args.output_dir, run_name+".log"), 'w') as fd:
    #     with contextlib.redirect_stderr(fd):
    #         run_it(hparams, args, run_name)
    run_it(hparams, args, run_name, pre_file_name)
    # --

def run_it(hparams, args, run_name, pre_file_name):
    # --
    file_handler = logging.FileHandler(os.path.join(args.output_dir, run_name+".log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOG.addHandler(file_handler)
    # --

    with open(args.path_test) as fd:
        data_test0 = json.load(fd)
        if args.ds_size is not None:
            data_test = data_test0[:args.ds_size]
        else:
            data_test = data_test0
        LOG.info(f"Load {len(data_test0)}=>{len(data_test)} from {args.path_test}")

    if args.path_train:
        with open(args.path_train) as fd3:
            data_train = json.load(fd3)
    else:
        data_train = None

    # -- overwrite those in hparams
    hparams.model_parallel = True  # note: allow init with gpu!
    if args.fp16 is not None:
        hparams.fp16 = bool(args.fp16)
    else:
        hparams.fp16 = False
    if args.batch_size is not None:
        hparams.batch_size = args.batch_size
    # --

    pre_file_dir = args.prefile_dir if args.prefile_dir else args.output_dir
    pre_file = os.path.join(pre_file_dir, f"fp16{int(hparams.fp16)}_{pre_file_name}.pre_edit.json")

    editor = MyEditor.from_hparams(hparams)
    # with torch.autocast(device_type="cuda"):  # this causes backward no-grad error?
    if 1:
        metrics, edited_model, _ = editor.edit(
            requests=data_test,
            keep_original_weight=True,
            edit_size=args.edit_size,
            pre_file=pre_file,
            train_insts=data_train,
        )

    summary = summary_metrics([z['post'] for z in metrics])
    LOG.info(f"Final result summary [L={len(metrics)}]: {summary}")
    json.dump(metrics, open(os.path.join(args.output_dir, run_name+".results.json"), 'w'), indent=4)

# --
if __name__ == "__main__":
    main()

"""
python -m mspx.tasks.me.run_me ...
"""
