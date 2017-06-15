
CUDA_VISIBLE_DEVICES="1" python train.py \
	--train_dir="../model/wide_and_deep" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--model=WideAndDeepModel \
	--wide_and_deep_models="FrameLevelLogisticModel,LstmMemoryModel" \
	--batch_size=128 \
	--base_learning_rate=0.0008 \
	--num_epochs=5 \
	--lstm_cells=1024 \
	--lstm_layers=2 \
	--moe_num_mixtures=8 \
	--rnn_swap_memory=True
