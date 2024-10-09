conda create -n fmfl python=3.7.13
conda activate fmfl
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt

mkdir vilt/pretrained_model_weight
cd vilt/pretrained_model_weight
wget https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt
cd ../..
mkdir datasets/missing_tables
bash scripts/training/run_hatememes.sh