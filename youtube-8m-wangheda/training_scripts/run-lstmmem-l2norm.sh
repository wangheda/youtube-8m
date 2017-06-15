
CUDA_VISIBLE_DEVICES=1 python train.py \
	--train_dir="../model/lstmmemory1024_moe8_l2norm/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--moe_num_mixtures=8 \
	--model=LstmMemoryNormalizationModel \
	--lstm_normalization="l2_normalize" \
	--feature_transformer="IdenticalTransformer" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--lstm_layers=2 \
	--lstm_cells=1024 \
	--base_learning_rate=0.0008 \
	--num_readers=1 \
	--rnn_swap_memory=True \
	--batch_size=128

