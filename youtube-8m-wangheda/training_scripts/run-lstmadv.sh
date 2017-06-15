
CUDA_VISIBLE_DEVICES="1" python train.py --train_dir="../model/lstmadvanced1024_moe2" --train_data_pattern="/Youtube-8M/data/frame/train/train*" --feature_names="rgb,audio" --feature_sizes="1024,128" --frame_features=True --model="LstmAdvancedModel" --batch_size=96 --base_learning_rate=0.0004 --num_epochs=5 --moe_num_mixtures=2 --lstm_cells=1024 --lstm_layers=1 --rnn_swap_memory=True
