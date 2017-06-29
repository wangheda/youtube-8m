
for part in ensemble_train ensemble_validate test; do 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/video_moe16_model" \
        --model_checkpoint_path="../model/video_moe16_model/model.ckpt-19058" \
	      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
	      --frame_features=False \
	      --feature_names="mean_rgb,mean_audio" \
	      --feature_sizes="1024,128" \
        --model=MoeModel \
        --moe_num_mixtures=16 \
	      --batch_size=128 \
	      --file_size=4096
done
