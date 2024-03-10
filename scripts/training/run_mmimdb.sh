python run.py with datasets=mmimdb \
        load_path='vilt/pretrained_model_weight/vilt_200k_mlm_itm.ckpt' \
        exp_name='mmimdb' \
        data_root='datasets/mmimdb' \
        num_gpus=1 \
        per_gpu_batchsize=1 \
        task_finetune_mmimdb \
        prompt_type=input 
