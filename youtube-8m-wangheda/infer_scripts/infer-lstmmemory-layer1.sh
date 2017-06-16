
#for part in ensemble_train ensemble_validate test; do 
for part in ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/lstmmemory1024_layer1_moe8" \
        --model_checkpoint_path="../model/lstmmemory1024_layer1_moe8/model.ckpt-149022" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
			  --model=LstmMemoryModel \
        --lstm_cells=1024 \
        --lstm_layers=1 \
        --moe_num_mixtures=8 \
        --rnn_swap_memory=False \
	      --batch_size=128 \
	      --file_size=4096
done
