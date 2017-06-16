
for part in ensemble_train ensemble_validate test; do 
  CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
    --output_dir="/Youtube-8M/model_predictions/${part}/distillchain_v2_lstmparalleloutput" \
    --model_checkpoint_path="../model/distillchain_v2_lstmparalleloutput/model.ckpt-74190" \
    --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord" \
    --distillation_features=False \
    --distillation_as_input=True \
    --model=DistillchainLstmParallelFinaloutputModel \
    --rnn_swap_memory=True \
    --lstm_cells="1024,128" \
    --moe_num_mixtures=4 \
    --batch_size=128 \
    --file_size=4096
done
