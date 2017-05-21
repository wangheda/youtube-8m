
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/video_bagging/base_model" \
	--train_data_pattern="/Youtube-8M/data/video/train/train*" \
	--moe_num_mixtures=4 \
	--model=DeepCombineChainModel \
	--label_loss=MultiTaskCrossEntropyLoss \
	--deep_chain_relu_cells=256 \
	--deep_chain_layers=4 \
	--frame_features=False \
	--feature_names="mean_rgb,mean_audio" \
	--feature_sizes="1024,128" \
	--multitask=True \
	--support_type="label,label,label,label" \
	--num_supports=18864 \
	--support_loss_percent=0.05 \
	--keep_checkpoint_every_n_hour=0.25 \
	--base_learning_rate=0.01 \
	--data_augmenter=NoiseAugmenter \
  --reweight=True \
  --sample_vocab_file=resources/train.video_id.vocab \
  --sample_freq_file=resources/train.video_id.freq.default \
	--input_noise_level=0.2 \
	--num_readers=2 \
	--num_epochs=4 \
	--batch_size=1024

