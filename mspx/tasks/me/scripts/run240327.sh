#

# run it

# --
# debug
EE_HOME=../EasyEdit/
MSPX_HOME=../src/
method=ICE
#CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -mpdb run_easy_edit.py ...
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -mpdb -m mspx.tasks.me.run_me --editing_method=$method --hparams_dir=${method}__Llama-2-7b-hf.yaml --path_test=../dataME/knowedit1.recent_test.json --path_train=../dataME/knowedit1.recent_train.json --output_dir=./ --prefile_dir=./ --ds_size 20

# --
# readings
#FT/LoRA: need the "target_new" mode to have reasonable loss!
#ROME: strange tok splitting & concating in compute_v (especially strange with llama)
#MEMIT: seems very similar, but there are still many places of differences: multiple-layers, prompts, solving-method

# --
# run more
EE_HOME=../EasyEdit/
MSPX_HOME=../src/
#RUN_NAME=run240330
RUN_NAME=run240402
#for method in FT LoRA IKE ROME; do
#for method in MEMIT; do
for method in ICE; do
for ff in knowedit1.recent_test.json:knowedit1.recent_train.json knowedit1.test_cf.json:knowedit1.train_cf.json knowedit1.ZsRE-test-all.json:zsre1.train_10000.json MQuAKE-CF-3k.json:MQuAKE-CF-5k.json MQuAKE-T.json:MQuAKE-CF-5k.json; do
IFS=: read -r w0 w1 <<< $ff
#echo "PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -u -m mspx.tasks.me.run_me --editing_method=$method --hparams_dir=${method}__Llama-2-7b-hf.yaml --path_test=../dataME/$w0 --path_train=../dataME/$w1 --output_dir=${RUN_NAME} >_log_${w0}_${method} 2>&1"
echo "PYTHONPATH=${EE_HOME}:${MSPX_HOME} python3 -u -m mspx.tasks.me.run_me --editing_method=$method --hparams_dir=${method}__Llama-2-7b-hf.yaml --path_test=../dataME/$w0 --path_train=../dataME/$w1 --output_dir=${RUN_NAME} --prefile_dir=run_prefiles >_log_${w0}_${method} 2>&1"
done
done >_cmds.${RUN_NAME}
python3 -m mspx.scripts.tools.run_para -f _cmds.${RUN_NAME} -n 100 -g 0 1 2 3 4 5 6 7 --default_num_gpus 2
# for ff in knowedit1.recent_test knowedit1.test_cf knowedit1.ZsRE-test-all MQuAKE-CF-3k MQuAKE-T; do grep -o "zres'.*" run240330/_log*${ff}*; done
# E=edit, R=rephrase, P=portability, L=locality || Three scores are: teacher_forcing.partial_acc || teacher_forcing.full_acc || greedy.prefix_matching_acc
# --
_log_knowedit1.recent_test.json_FT:zres': 'E=1.0000||1.0000||1.0000;;R=0.8937||0.7054||0.7212;;P=0.5077||0.1933||0.1983;;L=0.7545||0.4566||0.4263'}
_log_knowedit1.recent_test.json_IKE:zres': 'E=1.0000||1.0000||1.0000;;R=0.9266||0.8002||0.8002;;P=0.6912||0.3514||0.3763;;L=0.6014||0.2226||0.0031'}
_log_knowedit1.recent_test.json_LoRA:zres': 'E=0.9998||0.9976||0.9976;;R=0.9535||0.8641||0.8839;;P=0.5178||0.2038||0.2086;;L=0.5331||0.1778||0.1301'}
_log_knowedit1.recent_test.json_ROME:zres': 'E=0.9728||0.9139||0.9494;;R=0.8598||0.6406||0.6509;;P=0.4762||0.1272||0.1374;;L=0.5230||0.1823||0.1193'}
# --
_log_knowedit1.test_cf.json_FT:zres': 'E=1.0000||1.0000||1.0000;;R=0.9172||0.7187||0.7247;;P=0.6708||0.4590||0.4675;;L=0.8229||0.5596||0.5128'}
_log_knowedit1.test_cf.json_IKE:zres': 'E=0.9985||0.9940||0.9940;;R=0.9384||0.7652||0.7652;;P=0.8089||0.6074||0.6268;;L=0.6846||0.3099||0.0126'}
_log_knowedit1.test_cf.json_LoRA:zres': 'E=1.0000||1.0000||1.0000;;R=0.9586||0.8403||0.8641;;P=0.6934||0.5185||0.5282;;L=0.5620||0.2037||0.1391'}
_log_knowedit1.test_cf.json_ROME:zres': 'E=0.9872||0.9416||0.9595;;R=0.8290||0.5268||0.5292;;P=0.4935||0.2264||0.2326;;L=0.4791||0.1292||0.0700'}
# --
_log_knowedit1.ZsRE-test-all.json_FT:zres': 'E=1.0000||1.0000||1.0000;;R=0.9401||0.8132||0.8378;;P=0.6144||0.1176||0.1238;;L=0.9002||0.7202||0.6507'}
_log_knowedit1.ZsRE-test-all.json_IKE:zres': 'E=0.9997||0.9985||0.9985;;R=0.9992||0.9977||0.9977;;P=0.7814||0.3759||0.3759;;L=0.4942||0.0715||0.0380'}
_log_knowedit1.ZsRE-test-all.json_LoRA:zres': 'E=0.9998||0.9992||0.9992;;R=0.9908||0.9677||0.9708;;P=0.6046||0.1376||0.1384;;L=0.5693||0.2164||0.1434'}
_log_knowedit1.ZsRE-test-all.json_ROME:zres': 'E=0.9711||0.8493||0.8962;;R=0.9045||0.6779||0.7302;;P=0.5416||0.0331||0.0438;;L=0.4991||0.1418||0.0784'}
# --
_log_MQuAKE-CF-3k.json_FT:zres': 'E=1.0000||1.0000||1.0000;;R=NA||NA||NA;;P=0.5235||0.2142||0.2212;;L=NA||NA||NA'}
_log_MQuAKE-CF-3k.json_IKE:zres': 'E=1.0000||1.0000||1.0000;;R=NA||NA||NA;;P=0.7407||0.5962||0.5998;;L=NA||NA||NA'}
_log_MQuAKE-CF-3k.json_LoRA:zres': 'E=0.9985||0.9962||0.9962;;R=NA||NA||NA;;P=0.6463||0.4411||0.4481;;L=NA||NA||NA'}
_log_MQuAKE-CF-3k.json_ROME:zres': 'E=0.6154||0.4934||0.4934;;R=NA||NA||NA;;P=0.2335||0.0092||0.0118;;L=NA||NA||NA'}
# --
_log_MQuAKE-T.json_FT:zres': 'E=1.0000||1.0000||1.0000;;R=NA||NA||NA;;P=0.8841||0.6362||0.6711;;L=NA||NA||NA'}
_log_MQuAKE-T.json_IKE:zres': 'E=1.0000||1.0000||1.0000;;R=NA||NA||NA;;P=0.9347||0.7511||0.7511;;L=NA||NA||NA'}
_log_MQuAKE-T.json_LoRA:zres': 'E=1.0000||1.0000||1.0000;;R=NA||NA||NA;;P=0.8848||0.7024||0.7084;;L=NA||NA||NA'}
_log_MQuAKE-T.json_ROME:zres': 'E=1.0000||1.0000||1.0000;;R=NA||NA||NA;;P=0.5619||0.0005||0.0009;;L=NA||NA||NA'}
# --

# --
# use the same set of pre_file?
mkdir run_prefiles
for ff in knowedit1__recent_test knowedit1__test_cf knowedit1__ZsRE-test-all MQuAKE-CF-3k MQuAKE-T; do
  cp run240330/FT_llama-2-7b_${ff}__json.pre_edit.json run_prefiles/fp160_llama-2-7b_${ff}__json.pre_edit.json
  cp run240330/ROME_llama-2-7b_${ff}__json.pre_edit.json run_prefiles/fp161_llama-2-7b_${ff}__json.pre_edit.json
done
# --
