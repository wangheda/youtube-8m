
for part in test ensemble_train ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/lstm_positional_attention8max" \
        --model_checkpoint_path="../model/lstm_positional_attention8max/model.ckpt-198407" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
        --model=LstmPositionalAttentionMaxPoolingModel \
        --moe_num_mixtures=8 \
        --lstm_attentions=8 \
        --positional_embedding_size=32 \
        --rnn_swap_memory=True \
	      --batch_size=32 \
	      --file_size=4096
done
