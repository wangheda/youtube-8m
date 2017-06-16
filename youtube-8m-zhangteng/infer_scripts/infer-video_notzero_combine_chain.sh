
#for part in ensemble_train ensemble_validate test; do 
for part in test; do 
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/video_notzero_combine_chain" \
	      --model_checkpoint_path="../model/video_level_moemix4_model/model.ckpt-14668" \
	      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
	      --frame_features=False \
	      --feature_names="mean_rgb,mean_audio" \
	      --feature_sizes="1024,128" \
	      --batch_size=128 \
	      --file_size=4096
done
#for part in ensemble_train ensemble_validate test; do
for part in test; do
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
              --output_dir="/Youtube-8M/model_predictions/${part}/video_weight_combine_chain" \
              --model_checkpoint_path="../model/video_level_moemix4_weight_model/model.ckpt-17415" \
              --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
              --frame_features=False \
              --feature_names="mean_rgb,mean_audio" \
              --feature_sizes="1024,128" \
              --batch_size=128 \
              --file_size=4096
done
#for part in ensemble_train ensemble_validate test; do
for part in test; do
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
              --output_dir="/Youtube-8M/model_predictions/${part}/video_knowledge_combine_chain" \
              --model_checkpoint_path="../model/video_level_moeknowledge_model/model.ckpt-9606" \
              --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
              --frame_features=False \
              --feature_names="mean_rgb,mean_audio" \
              --feature_sizes="1024,128" \
              --batch_size=128 \
              --file_size=4096
done
#for part in ensemble_train ensemble_validate test; do
for part in test; do
    CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble.py \
              --output_dir="/Youtube-8M/model_predictions/${part}/video_softmax_combine_chain" \
              --model_checkpoint_path="../model/video_level_moesoftmax_model/model.ckpt-9501" \
              --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
              --frame_features=False \
              --feature_names="mean_rgb,mean_audio" \
              --feature_sizes="1024,128" \
              --batch_size=128 \
              --file_size=4096
done
                            
