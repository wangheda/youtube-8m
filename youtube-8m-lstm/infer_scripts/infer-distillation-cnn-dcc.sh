
for part in train; do 
  CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
    --output_dir="/Youtube-8M/model_predictions_local/${part}/distillation/distillchain_cnn_dcc" \
    --model_checkpoint_path="../model/distillation/distillchain_cnn_dcc/model.ckpt-74488" \
    --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_mean_model/*.tfrecord" \
    --distillation_features=True \
    --distillation_as_input=True \
    --model=DistillchainCnnDeepCombineChainModel \
    --deep_chain_layers=3 \
    --deep_chain_relu_cells=256 \
    --moe_num_mixtures=4 \
    --batch_size=128 \
    --file_size=4096
done
