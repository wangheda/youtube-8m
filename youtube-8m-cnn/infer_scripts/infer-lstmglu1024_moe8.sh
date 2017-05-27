
#for part in ensemble_train ensemble_validate test; do 
for part in ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/home/zhangt/yt8m/model_predictions/${part}/lstmglu_cell1024_layer1_moe8" \
              --model_checkpoint_path="/home/zhangt/yt8m/frame_level_lstm_glu_model/model.ckpt-111581" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
	      --batch_size=32 \
	      --file_size=4096
done
