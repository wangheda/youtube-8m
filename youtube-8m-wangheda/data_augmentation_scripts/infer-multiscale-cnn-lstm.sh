
CUDA_VISIBLE_DEVICES=1 python inference-rebuild.py \
	--output_file="../model/dataaugmentation_multiscale_cnn_lstm/predictions.772450" \
	--model_checkpoint_path="../model/dataaugmentation_multiscale_cnn_lstm/model.ckpt-772450" \
	--input_data_pattern="/Youtube-8M/data/frame/test/*.tfrecord" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--model='MultiscaleCnnLstmModel' \
	--multiscale_cnn_lstm_layers=4 \
	--moe_num_mixtures=4 \
	--is_training=False \
	--data_augmenter=HalfAugmenter \
	--batch_size=16 \
	--file_size=4096
