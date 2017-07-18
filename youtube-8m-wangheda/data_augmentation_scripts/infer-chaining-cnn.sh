
CUDA_VISIBLE_DEVICES=1 python inference-rebuild.py \
	--output_file="../model/dataaugmentation_chaining_cnn/predictions.1250814" \
	--model_checkpoint_path="../model/dataaugmentation_chaining_cnn/model.ckpt-1250814" \
	--input_data_pattern="/Youtube-8M/data/frame/test/*.tfrecord" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--model=CnnDeepCombineChainModel \
	--deep_chain_layers=3 \
	--deep_chain_relu_cells=128 \
	--moe_num_mixtures=4 \
	--data_augmenter=HalfAugmenter \
	--batch_size=16 \
	--file_size=4096
