
for part in ensemble_train ensemble_validate test; do 
#for part in ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-distill.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/distillchain_video_norm_moe8" \
          --model_checkpoint_path="../model/video_level_distillchainnorm2_model/model.ckpt-13412" \
	      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
	      --distill_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_mean_model/*.tfrecord" \
	      --frame_features=False \
	      --feature_names="mean_rgb,mean_audio" \
	      --distill_names="predictions" \
	      --feature_sizes="1024,128" \
	      --distill_sizes="4716" \
          --model=MoeDistillChainNorm2Model \
          --moe_num_extend=4 \
          --moe_method=None \
          --lstm_cells=1024 \
		  --lstm_layers=1 \
          --moe_num_mixtures=8 \
          --train=False \
	      --batch_size=128 \
	      --file_size=4096
done
