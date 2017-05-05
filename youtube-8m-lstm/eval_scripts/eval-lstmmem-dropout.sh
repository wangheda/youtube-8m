
GPU_ID=0
EVERY=1000
MODEL=LstmMemoryModel
MODEL_DIR="../model/lstmmemory1024_moe8_dropout"

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
        		--moe_num_mixtures=8 \
			--lstm_cells=1024 \
			--lstm_layers=2 \
			--dropout=True \
			--batch_size=64 \
			--run_once=True
	fi
done

