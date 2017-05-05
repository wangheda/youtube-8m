CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/lstmmemory1024_moe8_dropout/" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
        --batch_size=128 \
        --moe_num_mixtures=8 \
        --model=LstmMemoryModel \
        --num_readers=4 \
        --base_learning_rate=0.0005 \
	--keep_checkpoint_every_n_hours=1.0 \
        --lstm_cells=1024 \
        --lstm_layers=2 \
	--dropout=True \
	--keep_prob=0.8 \
        --rnn_swap_memory=True

