
CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/video_chain_moe4_deep_combine_addnoise/after_relu_noise_level_$1" \
        --frame_features=False \
        --model=DeepCombineChainModel \
        --label_loss=MultiTaskCrossEntropyLoss \
        --feature_names="mean_rgb,mean_audio" \
        --feature_sizes="1024,128" \
        --train_data_pattern="/Youtube-8M/data/video/train/train*" \
        --multitask=True \
        --support_type="label,label,label,label" \
        --support_loss_percent=0.05 \
        --support_size=18864 \
        --deep_chain_layers=4 \
        --deep_chain_relu_cells=256 \
        --moe_num_mixtures=4 \
        --noise_level=$1 \
        --keep_checkpoint_every_n_hours=0.25 \
        --base_learning_rate=0.01 \
        --num_readers=4 \
        --batch_size=1024 \
        --num_epochs=6 \
        --base_learning_rate=0.01

