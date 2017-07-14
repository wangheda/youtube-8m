
GPU_ID="0"
EVERY=4000
MODEL=CnnModel
MODEL_DIR="../model/cnn_model"

start=$1
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/frame/validate/validatea*" \
			--frame_features=True \
			--feature_names="rgb,audio" \
			--feature_sizes="1024,128" \
			--model=$MODEL \
      --cnn_num_filters=512 \
      --moe_num_mixtures=4 \
      --num_readers=4 \
      --batch_size=128 \
			--run_once=True
	fi
done

