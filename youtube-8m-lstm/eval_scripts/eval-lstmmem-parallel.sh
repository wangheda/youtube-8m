
start=$1
GPU_ID=0
MODEL=LstmParallelMemoryModel
MODEL_DIR="../model/lstmparallelmemory1024_moe8"

for checkpoint in $(for filename in $MODEL_DIR/model.ckpt-*.meta; do echo $filename | grep -o "ckpt-[0123456789]*.meta" | cut -d '-' -f 2 | cut -d '.' -f 1; done | sort -n); do
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/frame/validate/validatea*" \
			--frame_features=True \
			--feature_names="rgb,audio" \
			--feature_sizes="1024,128" \
			--lstm_cells="1024,128" \
			--batch_size=128 \
			--model=$MODEL \
			--moe_num_mixtures=8 \
			--rnn_swap_memory=True \
			--run_once=True
	fi
done

