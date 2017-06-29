CUDA_VISIBLE_DEVICES=0 python train.py \
		--train_data_pattern='/Youtube-8M/data/video/train/*.tfrecord' \
		--train_dir='../model/video_level_moeknowledge_model' \
		--model='MoeKnowledgeModel' \
		--feature_names="mean_rgb, mean_audio" \
		--feature_sizes="1024, 128" \
		--moe_num_mixtures=4 \
		--moe_layers=3 \
		--class_size=100
                            
