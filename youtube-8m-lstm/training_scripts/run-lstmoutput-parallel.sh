CUDA_VISIBLE_DEVICES=1 python train.py \
	--train_dir="../model/lstmparallelfinaloutput1024_moe8" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--batch_size=128 \
	--lstm_cells="1024,128" \
	--moe_num_mixtures=8 \
	--model=LstmParallelFinaloutputModel \
	--rnn_swap_memory=True \
	--num_readers=1 \
	--num_epochs=3 \
	--base_learning_rate=0.001

