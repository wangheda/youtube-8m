
CUDA_VISIBLE_DEVICES=0 python train.py --train_dir="../model/bilstm1024_layer2_moe2" --train_data_pattern="/Youtube-8M/data/frame/train/train*" --frame_features=True --model=BiLstmModel --batch_size=96 --base_learning_rate=0.0002
