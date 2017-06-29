
for part in ensemble_train ensemble_validate test; do 
#for part in ensemble_validate test; do 
    CUDA_VISIBLE_DEVICES=1 python inference_with_rebuild.py \
 	      --output_dir="/Youtube-8M/model_predictions/${part}/lstmglu2_cell1024_layer1_moe8" \
          --model_checkpoint_path="../model/frame_level_lstm_glu2_model/model.ckpt-132334" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
          --model=LstmGlu2Model \
          --video_level_classifier_model=MoeModel \
          --moe_num_extend=4 \
          --moe_method=None \
          --lstm_cells=1024 \
          --moe_num_mixtures=8 \
          --train=False \
	      --batch_size=32 \
	      --file_size=4096
done
