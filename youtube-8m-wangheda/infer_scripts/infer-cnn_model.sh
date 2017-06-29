
#for part in ensemble_train ensemble_validate test; do 
for part in ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/cnn_model" \
        --model_checkpoint_path="../model/cnn_model/model.ckpt-374098" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
        --model=CnnModel \
        --cnn_num_filters=512 \
        --moe_num_mixtures=4 \
	      --batch_size=128 \
	      --file_size=4096
done
