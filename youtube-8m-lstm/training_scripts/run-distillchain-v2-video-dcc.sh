noise_level=0.15

CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/distillchain_v2_video_dcc" \
	--train_data_pattern="/Youtube-8M/distillation_v2/video/train/*.tfrecord" \
	--distillation_features=True \
	--distillation_as_input=True \
	--frame_features=False \
	--feature_names="mean_rgb,mean_audio" \
	--feature_sizes="1024,128" \
	--model=DistillchainDeepCombineChainModel \
	--moe_num_mixtures=4 \
	--deep_chain_layers=4 \
	--deep_chain_relu_cells=256 \
	--data_augmenter=NoiseAugmenter \
	--input_noise_level=$noise_level \
	--multitask=True \
	--label_loss=MultiTaskCrossEntropyLoss \
	--support_type="label,label,label,label" \
	--num_supports=18864 \
	--support_loss_percent=0.05 \
	--base_learning_rate=0.007 \
	--keep_checkpoint_every_n_hour=0.25 \
	--num_readers=5 \
	--num_epochs=3 \
	--batch_size=512

