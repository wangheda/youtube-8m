for i in {1..3}; do
    for part in ensemble_train ensemble_validate test; do 
    #for part in train_samples; do 
        CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	          --output_dir="/Youtube-8M/model_predictions/${part}/lstm_random_moe8/sub_model_${i}" \
                  --model_checkpoint_path="/home/zhangt/yt8m/frame_level_lstm_random_model/model.ckpt-145850" \
	          --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	          --frame_features=True \
	          --feature_names="rgb,audio" \
	          --feature_sizes="1024,128" \
	          --batch_size=32 \
	          --file_size=4096
    done
done
