
for part in ensemble_train ensemble_validate test; do 
  output_dir="/Youtube-8M/model_predictions/${part}/distillchain_v2_boost_lstmparalleloutput"
  if [ ! -d $output_dir ]; then
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
        --output_dir="$output_dir" \
        --model_checkpoint_path="../model/distillchain_v2_boost_lstmparalleloutput/model.ckpt-86757" \
        --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
        --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --distillation_features=False \
        --distillation_as_input=True \
        --model=DistillchainLstmParallelFinaloutputModel \
        --rnn_swap_memory=True \
        --lstm_layers=1 \
        --lstm_cells="1024,128" \
        --moe_num_mixtures=8 \
        --batch_size=32 \
        --file_size=4096
  fi
done
