
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/framehop_lstm/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--model=FramehopLstmMemoryModel \
	--deep_chain_layers=4 \
	--deep_chain_relu_cells=256 \
	--moe_num_mixtures=4 \
	--keep_checkpoint_every_n_hours=1.0 \
	--base_learning_rate=0.0008 \
	--feature_transformer=IdenticalTransformer \
	--num_readers=4 \
	--batch_size=128 \
	--num_epochs=5 \
	--rnn_swap_memory=True \
	--lstm_layers=2 \
	--lstm_cells="512,64"

