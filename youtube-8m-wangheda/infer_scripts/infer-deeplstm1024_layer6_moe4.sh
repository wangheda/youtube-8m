
for part in test ensemble_train ensemble_validate train_samples; do 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/deeplstm1024_layer6_moe4" \
        --model_checkpoint_path="../model/deeplstm1024_layer6_moe4/model.ckpt-175048" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
	      --batch_size=128 \
	      --file_size=4096
done
