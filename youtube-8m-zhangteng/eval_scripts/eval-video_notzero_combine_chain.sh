GPU_ID=0
EVERY=1000
MODEL=MoeMix4Model
MODEL_DIR="../model/video_level_moemix4_model"
start=0
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [[ $checkpoint -gt $start ]]; then

		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/video/validate/validatea*" \
			--frame_features=False \
			--feature_names="mean_rgb,mean_audio" \
			--feature_sizes="1024,128" \
			--batch_size=128 \
			--model=$MODEL \
			--class_size=100 \
			--moe_num_mixtures=4 \
			--moe_layers=3 \
			--run_once=True
	fi
done

                            
