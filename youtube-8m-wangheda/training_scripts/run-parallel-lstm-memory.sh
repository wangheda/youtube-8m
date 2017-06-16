
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/lstmparallelmemory1024_moe8" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--batch_size=128 \
	--lstm_cells="1024,128" \
	--moe_num_mixtures=8 \
	--model=LstmParallelMemoryModel \
	--rnn_swap_memory=True \
	--num_readers=4 \
	--num_epochs=5 \
	--base_learning_rate=0.0008
