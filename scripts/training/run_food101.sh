python run.py with datasets=Food101 \
        load_path='vilt/pretrained_model_weight/vilt_200k_mlm_itm.ckpt' \
        exp_name='food101' \
        data_root='datasets/Food101' \
        num_gpus=1 \
        max_text_len=40 \
        per_gpu_batchsize=88 \
        num_workers=0 \
        batch_size=512 \
        task_finetune_food101 \
        prompt_type=input  