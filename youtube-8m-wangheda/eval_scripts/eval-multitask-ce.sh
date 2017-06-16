GPU_ID=1
EVERY=2000
MODEL=LstmMemoryMultitaskModel
MODEL_DIR="../model/lstmmemory1024_moe8_multitask_ce"

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
			--batch_size=32 \
			--model=$MODEL \
			--lstm_layers=2 \
			--lstm_cells=1024 \
			--multitask=True \
			--moe_num_mixtures=8 \
			--num_supports=25 \
			--num_verticals=25 \
			--vertical_file="resources/vertical.tsv" \
			--run_once=True
	fi
done

