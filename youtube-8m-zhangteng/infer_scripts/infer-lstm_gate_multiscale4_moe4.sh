
for part in ensemble_train ensemble_validate test; do 
#for part in ensemble_validate test; do 
    CUDA_VISIBLE_DEVICES=0 python inference_with_rebuild.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/lstm_gate_multiscale4_moe4" \
          --model_checkpoint_path="../model/frame_level_lstm_multiscale2_model/model.ckpt-193803" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
          --model=LstmMultiscale2Model \
          --video_level_classifier_model=MoeModel \
          --moe_num_extend=4 \
          --moe_method=None \
          --lstm_cells=1024 \
          --moe_num_mixtures=4 \
	  --norm=False \
          --train=False \
	      --batch_size=32 \
	      --file_size=4096
done
