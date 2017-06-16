
#for part in ensemble_train ensemble_validate test; do 
for part in train_samples; do 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/video_moe4_noise0.2_layer4_elu" \
        --model_checkpoint_path="/home/zhangt/yt8m/video_level_moenoise0.2_model/moe_4layers_elu/model.ckpt-22845" \
	      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
	      --frame_features=False \
	      --feature_names="mean_rgb,mean_audio" \
	      --feature_sizes="1024,128" \
	      --batch_size=128 \
	      --file_size=4096
done
