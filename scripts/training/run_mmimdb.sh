CUDA_VISIBLE_DEVICES=2,3 python run.py with datasets=mmimdb \
        load_path='vilt/pretrained_model_weight/vilt_200k_mlm_itm.ckpt' \
        exp_name='mmimdb' \
        data_root='datasets/mmimdb' \
        num_gpus=2 \
        per_gpu_batchsize=88 \
        num_workers=0 \
        batch_size=512 \
        task_finetune_mmimdb \
        prompt_type=input 
