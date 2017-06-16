

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --output_file="../model/audio_lstmmemory1024_layer1_moe8/error_analysis.train_samples.tsv" \
    --model_checkpoint_path="../model/audio_lstmmemory1024_layer1_moe8/model.ckpt-187979" \
    --input_data_pattern="/Youtube-8M/data/frame/train/train1r.tfrecord" \
    --frame_features=True \
    --feature_names="audio" \
    --feature_sizes="128" \
    --batch_size=128 \
    --file_size=4096
