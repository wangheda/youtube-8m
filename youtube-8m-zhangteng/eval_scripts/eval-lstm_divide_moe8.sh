
GPU_ID=1
EVERY=1000
MODEL=LstmDivideModel
MODEL_DIR="/home/zhangt/yt8m/frame_level_lstm_divide_model"
start=80000
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
 	echo $checkpoint;
	if [[ $checkpoint -gt $start ]]; then

 		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
 			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/frame/validate/validatea*" \
			--frame_features=True \
			--feature_names="rgb,audio" \
			--feature_sizes="1024,128" \
			--batch_size=64 \
			--model=$MODEL \
			--video_level_classifier_model=MoeModel \
			--moe_num_extend=3 \
			--moe_num_mixtures=8 \
			--train=False \
			--run_once=True
	fi
done

