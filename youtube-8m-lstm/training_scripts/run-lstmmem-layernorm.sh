
CUDA_VISIBLE_DEVICES=1 python train.py \
	--train_dir="../model/layernormlstmmemory1024_moe8/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--model=LayerNormLstmMemoryModel \
	--moe_num_mixtures=8 \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--lstm_layers=2 \
	--lstm_cells=1024 \
	--base_learning_rate=0.0005 \
	--num_readers=2 \
	--rnn_swap_memory=True \
	--batch_size=128

