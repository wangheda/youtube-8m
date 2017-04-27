
#for part in ensemble_train ensemble_validate test; do 
for part in train_samples; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
          --output_dir="/Youtube-8M/model_predictions/${part}/lstmmem1024_layer2_moe4_deep_combine_chain_add_length" \
          --model_checkpoint_path="../model/lstmmem1024_deep_combine_chain_length/model.ckpt-148035" \
          --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
          --frame_features=True \
          --feature_names="rgb,audio" \
          --feature_sizes="1024,128" \
          --batch_size=32 \
          --file_size=4096
done
