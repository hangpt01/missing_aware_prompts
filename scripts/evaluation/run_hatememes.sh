python run.py with datasets=Hateful_Memes \
        load_path='vilt/pretrained_model_weight/vilt_200k_mlm_itm.ckpt' \
        exp_name='hateful_memes' \
        data_root='datasets/Hateful_Memes' \
        num_gpus=1 \
        max_text_len=40 \
        per_gpu_batchsize=1 \
        task_finetune_hatememes \
        prompt_type=input \
        test_only=True   