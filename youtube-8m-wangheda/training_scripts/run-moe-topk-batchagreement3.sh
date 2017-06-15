CUDA_VISIBLE_DEVICES=0 python train.py \
        --model=MoeModel \
        --moe_num_mixtures=16 \
        --label_loss=TopKBatchAgreementCrossEntropyLoss \
        --train_data_pattern="/Youtube-8M/data/video/train/train*" \
        --feature_names="mean_rgb,mean_audio" \
        --feature_sizes="1024,128" \
        --train_dir="../model/video_moe16_topk_batchagreement3" \
        --base_learning_rate=0.01 \
        --batch_agreement=3 \
        --keep_checkpoint_every_n_hours=0.25 \
        --num_epochs=5 \
        --batch_size=1024

