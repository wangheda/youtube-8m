
for part in ensemble_train ensemble_validate test; do 
  CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
    --output_dir="/Youtube-8M/model_predictions/${part}/distillchain_lstmparalleloutput" \
    --model_checkpoint_path="../model/distillchain_lstmparalleloutput/model.ckpt-75261" \
    --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_mean_model/*.tfrecord" \
    --distillation_features=True \
    --distillation_as_input=True \
    --model=DistillchainLstmParallelFinaloutputModel \
    --rnn_swap_memory=True \
    --lstm_cells="1024,128" \
    --moe_num_mixtures=8 \
    --batch_size=128 \
    --file_size=4096
done
