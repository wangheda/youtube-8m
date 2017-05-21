
CUDA_VISIBLE_DEVICES=0 python train.py --train_dir="../model/lstm1024_moe4/" --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --train_data_pattern="/Youtube-8M/data/frame/train/train*" --batch_size=96 --lstm_cells=1024 --moe_num_mixtures=4 --model=LstmModel --num_readers=4 --rnn_swap_memory=True --num_epochs=5 --base_learning_rate=0.0008
