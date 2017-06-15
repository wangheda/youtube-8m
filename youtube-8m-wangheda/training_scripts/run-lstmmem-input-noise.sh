
noise_level=$1 

python train.py \
	--train_dir="../model/lstmmemory1024_moe8_input_noise/noise_level_$1/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--model=LstmMemoryModel \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--lstm_layers=2 \
	--lstm_cells=1024 \
	--moe_num_mixtures=8 \
	--data_augmenter=NoiseAugmenter \
	--input_noise_level=$noise_level \
	--base_learning_rate=0.0008 \
	--rnn_swap_memory=True \
	--batch_size=128
