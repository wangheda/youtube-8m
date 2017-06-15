
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/video_deep_chain_moe4" \
	--train_data_pattern="/Youtube-8M/data/video/train/train*" \
	--moe_num_mixtures=4 \
	--model=DeepChainModel \
	--label_loss=MultiTaskCrossEntropyLoss \
	--deep_chain_relu_cells=200 \
	--deep_chain_layers=3 \
	--frame_features=False \
	--feature_names="mean_rgb,mean_audio" \
	--feature_sizes="1024,128" \
	--multitask=True \
	--support_type="label,label,label" \
	--num_supports=14148 \
	--support_loss_percent=0.1 \
	--keep_checkpoint_every_n_hour=0.25 \
	--base_learning_rate=0.01 \
	--num_readers=2 \
	--num_epochs=8 \
	--batch_size=1024

