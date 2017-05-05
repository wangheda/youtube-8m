
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/lstmmemory1024_moe8_mlp/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--model=LstmMemoryModel \
	--video_level_classifier_model=MlpMoeModel \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--moe_num_mixtures=8 \
	--lstm_layers=2 \
	--lstm_cells=1024 \
	--base_learning_rate=0.001 \
	--rnn_swap_memory=True \
	--batch_size=128
