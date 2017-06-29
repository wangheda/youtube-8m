
CUDA_VISIBLE_DEVICES=1 python train.py \
	--train_dir="../model/dataaugmentation_parallel_lstm_memory" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--train_data_pattern="/Youtube-8M/data/frame/largetrain/*.tfrecord" \
	--lstm_cells="1024,128" \
	--moe_num_mixtures=8 \
	--model=LstmParallelMemoryModel \
	--rnn_swap_memory=True \
	--num_readers=4 \
	--batch_size=40 \
        --data_augmenter=HalfAugmenter \
	--num_epochs=5 \
	--base_learning_rate=0.0008 \
    	--keep_checkpoint_every_n_hour=2.0
