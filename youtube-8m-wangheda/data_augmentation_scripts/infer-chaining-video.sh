
CUDA_VISIBLE_DEVICES=1 python inference-rebuild.py \
	--output_file="../model/dataaugmentation_chaining_video/predictions.125208" \
	--model_checkpoint_path="../model/dataaugmentation_chaining_video/model.ckpt-125208" \
	--input_data_pattern="/Youtube-8M/data/frame/test/*.tfrecord" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
        --model=DeepCombineChainModel \
        --moe_num_mixtures=2 \
        --deep_chain_layers=8 \
        --deep_chain_relu_cells=40 \
	--data_augmenter=HalfVideoAugmenter \
	--batch_size=64 
