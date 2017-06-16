
CUDA_VISIBLE_DEVICES=1 python train.py \
		--train_data_pattern='/Youtube-8M/data/frame/train/*.tfrecord' \
		--train_dir='../model/frame_level_lstm_extend8_model' \
		--frame_features=True \
		--feature_names="rgb, audio" \
		--feature_sizes="1024, 128" \
		--model='LstmExtendModel' \
		--video_level_classifier_model=MoeExtendModel \
		--moe_num_extend=8 \
		--moe_num_mixtures=8 \
		--batch_size=128 \
		--base_learning_rate=0.001
