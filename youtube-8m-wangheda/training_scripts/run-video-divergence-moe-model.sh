CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dir="../model/video_divergence_moe" \
    --train_data_pattern="/Youtube-8M/data/video/train/train*" \
    --frame_features=False \
    --feature_names="mean_rgb,mean_audio" \
    --feature_sizes="1024,128" \
    --model=MultiTaskDivergenceMoeModel \
    --divergence_model_count=8 \
    --moe_num_mixtures=4 \
    --multitask=True \
    --label_loss=MultiTaskDivergenceCrossEntropyLoss \
    --support_loss_percent=0.025 \
    --batch_size=256 \
    --num_readers=4 \
    --base_learning_rate=0.01 \
    --num_epochs=5

