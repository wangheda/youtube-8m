
for part in test ensemble_train ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble-get-input.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/model_input" \
	      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
	      --frame_features=False \
	      --feature_names="mean_rgb,mean_audio" \
	      --feature_sizes="1024,128" \
	      --batch_size=128 \
	      --file_size=4096
done
