#

# run it (again)

# --
# debug
EE_HOME=../EasyEdit/
MSPX_HOME=../src/
method=ICE
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -mpdb run_easy_edit.py ...
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -mpdb -m mspx.tasks.me.run_me --editing_method=$method --hparams_dir=${method}__Llama-2-7b-hf.yaml --path_test=../dataME/knowedit1.ZsRE-test-all.json --path_train=../dataME/zsre1.train_10000.json --output_dir=./ --prefile_dir=./ --data_template qa --ds_size 20

# --
# run more
EE_HOME=../EasyEdit/
MSPX_HOME=../src/
RUN_NAME=run240409
for method in FT LoRA IKE ROME MEMIT ICE; do
for ff in knowedit1.recent_test.json:knowedit1.recent_train.json:orig knowedit1.test_cf.json:knowedit1.train_cf.json:orig knowedit1.ZsRE-test-all.json:zsre1.train_10000.json:qa MQuAKE-CF-3k.json:MQuAKE-CF-5k.json:qa MQuAKE-T.json:MQuAKE-CF-5k.json:qa; do
IFS=: read -r w0 w1 wt <<< $ff
echo "PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -u -m mspx.tasks.me.run_me --editing_method=$method --hparams_dir=${method}__Llama-2-7b-hf.yaml --path_test=../dataME/$w0 --path_train=../dataME/$w1 --output_dir=${RUN_NAME} --prefile_dir=run_prefiles --data_template=$wt >_log_${w0}_${method} 2>&1"
done
done >_cmds.${RUN_NAME}
python3 -m mspx.scripts.tools.run_para -f _cmds.${RUN_NAME} -n 100 -g 0 1 2 3 4 5 6 7 --default_num_gpus 2
