
CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/dataaugmentation_chaining_cnn/" \
        --train_data_pattern="/Youtube-8M/data/frame/largetrain/*.tfrecord" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --model=CnnDeepCombineChainModel \
        --deep_chain_layers=4 \
        --deep_chain_relu_cells=128 \
        --moe_num_mixtures=4 \
        --multitask=True \
        --label_loss=MultiTaskCrossEntropyLoss \
        --support_type="label,label,label,label" \
        --support_loss_percent=0.05 \
        --num_readers=4 \
        --batch_size=40 \
        --data_augmenter=HalfAugmenter \
        --keep_checkpoint_every_n_hours=1.0 \
        --base_learning_rate=0.001

