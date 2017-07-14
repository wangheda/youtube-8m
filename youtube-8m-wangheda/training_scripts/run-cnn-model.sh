CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/cnn_model/" \
        --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --model=CnnModel \
        --cnn_num_filters=512 \
        --moe_num_mixtures=4 \
        --num_readers=4 \
        --batch_size=128 \
        --keep_checkpoint_every_n_hours=0.5 \
        --base_learning_rate=0.001

