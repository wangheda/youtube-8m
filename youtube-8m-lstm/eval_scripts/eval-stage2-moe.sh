
start=$1
GPU_ID=0
MODEL=MoeModel
MODEL_DIR="../model/video_moe16_stage2_moe8"

for checkpoint in $(for filename in $MODEL_DIR/model.ckpt-*.meta; do echo $filename | grep -o "ckpt-[0123456789]*.meta" | cut -d '-' -f 2 | cut -d '.' -f 1; done | sort -n); do
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/video/validate-validate-part1/validatea*" \
			--frame_features=False \
			--feature_names="predictions" \
			--feature_sizes=4716 \
			--batch_size=1024 \
                        --moe_num_mixtures=8 \
			--model=$MODEL \
			--run_once=True
	fi
done

