CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dir="../model/lstmlookback1024_moe8" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
    --batch_size=128 \
    --lstm_layers=2 \
    --lstm_cells="1024,128" \
    --moe_num_mixtures=8 \
    --model=LstmLookBackModel \
    --lstm_look_back=3 \
    --rnn_swap_memory=True \
    --num_readers=1 \
    --num_epochs=5 \
    --base_learning_rate=0.0008

