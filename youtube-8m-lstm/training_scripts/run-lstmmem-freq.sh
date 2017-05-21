
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/lstmmemory1024_moe8_chain_freq/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--moe_num_mixtures=8 \
	--model=LstmMemoryModel \
	--video_level_classifier_model=ChainSupportReluMoeModel \
	--label_loss=MultiTaskCrossEntropyLoss \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--lstm_layers=2 \
	--lstm_cells=1024 \
	--multitask=True \
	--support_type="frequent" \
	--num_supports=200 \
	--num_frequents=200 \
	--support_loss_percent=0.1 \
	--vertical_file="resources/vertical.tsv" \
	--base_learning_rate=0.0008 \
	--num_readers=2 \
	--batch_size=128

