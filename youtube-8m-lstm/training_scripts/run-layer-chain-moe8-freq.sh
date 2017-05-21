CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/layer_chain_moe8_freq/" \
        --frame_features=False \
        --model=ChainSupportReluMoeModel \
        --label_loss=MultiTaskCrossEntropyLoss \
        --feature_names="layer" \
        --feature_sizes="2048" \
        --train_data_pattern="/Youtube-8M/data/layer/train/train*" \
        --batch_size=1024 \
        --multitask=True \
        --num_supports=200 \
        --support_type=frequent \
        --support_loss_percent=0.1 \
        --moe_num_mixtures=8 \
        --keep_checkpoint_every_n_hours=0.25 \
        --num_readers=4 \
        --base_learning_rate=0.01

