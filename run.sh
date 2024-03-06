python run.py with data_root=datasets/Hateful_Memes \
        num_gpus=1 \
        # num_nodes=16 \
        per_gpu_batchsize=1 \
        task_finetune_hatememes \
        load_path=vilt/pretrained_model_weight \
        exp_name=test_hateful_memes \
        prompt_type=input \
        # test_ratio=<TEST_RATIO> \
        # test_type=<TEST_TYPE> \
        test_only=True   