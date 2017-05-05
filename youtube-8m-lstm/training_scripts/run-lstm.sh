
CUDA_VISIBLE_DEVICES=0 python train.py  --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --train_data_pattern="/Youtube-8M/data/frame/train/train*" --model=LstmModel --num_readers=1 --base_learning_rate=0.0008 --rnn_swap_memory=True --train_dir="../model/lstm1024_moe8" --moe_num_mixtures=8 --batch_size=128
