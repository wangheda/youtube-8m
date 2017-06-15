# for distillation version 2 data

cd ../youtube-8m-zhangteng

#for part in ensemble_train ensemble_validate test; do 
for part in train; do 
  output_dir="/Youtube-8M/model_predictions_x32/${part}/distillation/distillchain_lstm_gate_moe8"
  if [ ! -d $output_dir ]; then
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble-distill.py \
        --output_dir="$output_dir" \
        --model_checkpoint_path="../model/distillation/frame_level_lstm_gate_distillchain_model/model.ckpt-63344" \
        --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
        --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_mean_model/*.tfrecord" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --distill_names="predictions" \
        --feature_sizes="1024,128" \
        --distill_sizes="4716" \
        --model=LstmGateModel \
        --video_level_classifier_model=MoeDistillChainModel \
        --moe_num_extend=4 \
        --moe_method=None \
        --lstm_cells=1024 \
        --moe_num_mixtures=8 \
        --train=False \
        --batch_size=32 \
        --file_size=4096
  fi
done
