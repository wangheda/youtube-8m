CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/lstmmemory2048_moe4_batch128/" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
        --batch_size=128 \
        --moe_num_mixtures=4 \
        --model=LstmMemoryModel \
        --num_readers=4 \
        --base_learning_rate=0.0004 \
	--keep_checkpoint_every_n_hours=3.0 \
        --lstm_cells=2048 \
        --lstm_layers=2 \
        --rnn_swap_memory=True

