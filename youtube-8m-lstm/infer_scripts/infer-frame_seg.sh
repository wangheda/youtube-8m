
for part in ensemble_train ensemble_validate test; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions_local/${part}/frame_seg_model" \
        --model_checkpoint_path="../model/frame_seg_model/model.ckpt-27945" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
        --feature_transformer=IdenticalTransformer \
        --model=FrameSegModel \
        --moe_num_mixtures=16 \
	      --batch_size=64 \
	      --file_size=4096
done
