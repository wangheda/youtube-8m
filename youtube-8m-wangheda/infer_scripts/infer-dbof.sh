
for part in test ensemble_train ensemble_validate; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/dbof_model" \
        --model_checkpoint_path="../model/dbof_model/model.ckpt-184058" \
	      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
	      --frame_features=True \
	      --feature_names="rgb,audio" \
	      --feature_sizes="1024,128" \
	      --model=DbofModel \
	      --batch_size=64 \
        --num_readers=1 \
	      --file_size=4096
done
