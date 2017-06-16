
for part in ensemble_train ensemble_validate test; do 
#for part in ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble-distill.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/distillchain_lstm_moe8" \
          --model_checkpoint_path="../model/frame_level_lstm_distillchain_model/model.ckpt-49873" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_mean_model/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --distill_names="predictions" \
	      --feature_sizes="1024,128" \
	      --distill_sizes="4716" \
          --model=LstmModel \
          --video_level_classifier_model=MoeDistillChainModel \
          --moe_num_extend=4 \
          --moe_method=None \
          --lstm_cells=1024 \
		  --lstm_layers=1 \
          --moe_num_mixtures=8 \
          --train=False \
	      --batch_size=32 \
	      --file_size=4096
done
