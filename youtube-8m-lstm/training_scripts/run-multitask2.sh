
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/lstmmemory1024_moe8_multitask_ce/" \
	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
	--moe_num_mixtures=8 \
	--model=LstmMemoryMultitaskModel \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--lstm_layers=2 \
	--lstm_cells=1024 \
	--multitask=True \
	--num_verticals=25 \
	--vertical_file="resources/vertical.tsv" \
	--label_loss=MultiTaskCrossEntropyLoss \
	--base_learning_rate=0.0004 \
	--vertical_loss_percent=0.1 \
	--batch_size=96
