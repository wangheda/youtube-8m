percent="$1"
start=0

GPU_ID=0
EVERY=200
MODEL=DeepCombineChainModel
MODEL_DIR="../model/distillation_video_dcc_noise/scene2_percent_${percent}"

DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/video/validate/validatea*" \
			--frame_features=False \
			--feature_names="mean_rgb,mean_audio" \
			--feature_sizes="1024,128" \
			--model=$MODEL \
			--deep_chain_relu_cells=256 \
			--deep_chain_layers=4 \
			--moe_num_mixtures=4 \
			--batch_size=1024 \
			--num_readers=1 \
			--run_once=True
	fi
done

