
for part in ensemble_train ensemble_validate test; do 
#for part in ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-distill.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/distillchain_lstm_multiscale_layer4_moe8" \
          --model_checkpoint_path="../model/frame_level_lstm_multiscale_distillchain4_model/model.ckpt-79645" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_mean_model/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --distill_names="predictions" \
	      --feature_sizes="1024,128" \
	      --distill_sizes="4716" \
          --model=LstmMultiscaleDitillChainModel \
          --moe_num_extend=4 \
          --moe_method=None \
          --lstm_cells=1024 \
          --moe_num_mixtures=4 \
          --cnn_cells=196 \
          --train=False \
	      --batch_size=32 \
	      --file_size=4096
done
