
CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dir="../model/dataaugmentation_attention_pooling" \
    --train_data_pattern="/Youtube-8M/data/frame/largetrain/*.tfrecord" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --model=LstmPositionalAttentionMaxPoolingModel \
    --moe_num_mixtures=8 \
    --lstm_attentions=1 \
    --positional_embedding_size=32 \
    --rnn_swap_memory=True \
    --base_learning_rate=0.001 \
    --num_readers=4 \
    --num_epochs=6 \
    --batch_size=40 \
    --data_augmenter=HalfAugmenter \
    --keep_checkpoint_every_n_hour=2.0
