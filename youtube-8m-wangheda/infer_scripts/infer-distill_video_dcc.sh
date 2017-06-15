
for part in ensemble_train ensemble_validate test; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/distillation_video_dcc_noise" \
        --model_checkpoint_path="../model/distillation_video_dcc_noise/scene1_percent_0.4/model.ckpt-47842" \
	      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
	      --frame_features=False \
	      --feature_names="mean_rgb,mean_audio" \
	      --feature_sizes="1024,128" \
        --model=DeepCombineChainModel \
        --deep_chain_relu_cells=256 \
        --deep_chain_layers=4 \
        --moe_num_mixtures=4 \
	      --batch_size=32 \
	      --file_size=4096
done
