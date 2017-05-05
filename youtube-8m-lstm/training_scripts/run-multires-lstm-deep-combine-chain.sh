
CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_dir="../model/multires_lstm_deep_combine_chain/" \
        --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --label_loss=MultiTaskCrossEntropyLoss \
        --multitask=True \
        --support_type="label,label,label,label" \
        --support_loss_percent=0.05 \
        --support_size=18864 \
        --model=MultiresLstmMemoryDeepCombineChainModel \
        --deep_chain_layers=4 \
        --deep_chain_relu_cells=256 \
        --moe_num_mixtures=4 \
        --dropout=True \
        --keep_prob=0.9 \
        --keep_checkpoint_every_n_hours=2.0 \
        --base_learning_rate=0.0008 \
        --feature_transformer=IdenticalTransformer \
        --num_readers=4 \
        --batch_size=128 \
        --num_epochs=5 \
        --rnn_swap_memory=True \
        --lstm_cells="512,64" \
        --lstm_layers=2

