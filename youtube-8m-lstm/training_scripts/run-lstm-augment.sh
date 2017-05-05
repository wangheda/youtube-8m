
CUDA_VISIBLE_DEVICES=1 python train.py \
        --train_dir="../model/lstmmemory1024_moe8_augment/" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
        --batch_size=64 \
        --moe_num_mixtures=8 \
        --model=LstmMemoryModel \
        --data_augmenter=ClippingAugmenter \
        --num_readers=4 \
        --base_learning_rate=0.0004 \
	--keep_checkpoint_every_n_hours=1.5 \
        --lstm_cells=1024 \
        --lstm_layers=2 \
        --rnn_swap_memory=True

