#

# prepare data for editing

# --
# helpers
function download_gdrive_zip_file {
    ggID=$1
    archive=$2
    ggURL='https://drive.google.com/uc?export=download'
    echo "Downloading ${archive}"
    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${archive}"
}

# --
# first get vraious data; @dataME/

# from EasyEdit: https://github.com/zjunlp/EasyEdit?tab=readme-ov-file#datasets-for-factual-knowledge
download_gdrive_zip_file "1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4" editing-data.zip
unzip editing-data.zip
rm -rf __MACOSX/
mv data data_editing

# from ROME: https://rome.baulab.info
mkdir data_editing2
for ff in counterfact.json zsre_mend_eval.json zsre_mend_train.json; do
wget https://rome.baulab.info/data/dsets/$ff -O data_editing2/$ff
done

# from KnowEdit: https://huggingface.co/datasets/zjunlp/KnowEdit
git lfs install
git clone git@hf.co:datasets/zjunlp/KnowEdit
#git clone https://www.modelscope.cn/datasets/zjunlp/KnowEdit.git

# --
# download google drives
download_gdrive_zip_file "1IGt7NNV-OxXqIljjr02_k0dDY50Z5N_E" memit.data
