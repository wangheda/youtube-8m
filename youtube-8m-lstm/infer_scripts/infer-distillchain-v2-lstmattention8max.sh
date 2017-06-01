
for part in ensemble_train ensemble_validate test; do 
  output_dir="/Youtube-8M/model_predictions/${part}/distillchain_v2_lstmattention8max"
  if [ ! -d $output_dir ]; then
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
        --output_dir="$output_dir" \
        --model_checkpoint_path="../model/distillchain_v2_lstmattention8max/model.ckpt-" \
        --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
        --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --distillation_features=False \
        --distillation_as_input=True \
        --model=DistillchainLstmAttentionMaxPoolingModel \
        --moe_num_mixtures=8 \
        --lstm_attentions=8 \
        --lstm_cells=1024 \
        --rnn_swap_memory=True \
        --moe_num_mixtures=8 \
        --batch_size=128 \
        --file_size=4096
  fi
done
