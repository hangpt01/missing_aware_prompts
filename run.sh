CUDA_VISIBLE_DEVICES=0 python run.py with datasets=Hateful_Memes \
        load_path='vilt/pretrained_model_weight/vilt_200k_mlm_itm.ckpt' \
        exp_name='hateful_memes' \
        data_root='datasets/Hateful_Memes' \
        num_gpus=1 \
        max_text_len=40 \
        per_gpu_batchsize=1 \
        task_finetune_hatememes \
        prompt_type=input \
        test_only=True   
# python run.py with data_root=datasets/mmimdb \
#         num_gpus=1 \
#         # num_nodes=16 \
#         per_gpu_batchsize=1 \
#         task_finetune_mmimdb=1 \
#         load_path=vilt/pretrained_model_weight \
#         exp_name=test_mmimdb \
#         prompt_type=input \
#         # test_ratio=<TEST_RATIO> \
#         # test_type=<TEST_TYPE> \
#         test_only=True   