#!/bin/bash

data_path=/Youtube-8M/model_predictions_local/train

input_data_pattern="/Youtube-8M/data/video/train/*.tfrecord"
prediction_data_pattern="${data_path}/distillation/ensemble_v2_matrix_model/prediction*.tfrecord"

CUDA_VISIBLE_DEVICES="" python inference-combine-tfrecords-video.py \
	      --output_dir="/Youtube-8M/distillation_v2/video/train" \
        --input_data_pattern="$input_data_pattern" \
        --prediction_data_pattern="$prediction_data_pattern" \
	      --batch_size=1024 \
	      --file_size=4096
