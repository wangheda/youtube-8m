
start=$1
GPU_ID=0
MODEL=DistillchainDeepCombineChainModel
MODEL_DIR="../model/distillchain_video_dcc"

for checkpoint in $(for filename in $MODEL_DIR/model.ckpt-*.meta; do echo $filename | grep -o "ckpt-[0123456789]*.meta" | cut -d '-' -f 2 | cut -d '.' -f 1; done | sort -n); do
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/video/validate/validatea*" \
			--distill_data_pattern="/Youtube-8M/model_predictions/validatea/distillation/ensemble_mean_model/*.tfrecord" \
      --distillation_features=True \
      --distillation_as_input=True \
			--frame_features=False \
			--feature_names="mean_rgb,mean_audio" \
			--feature_sizes="1024,128" \
			--model=$MODEL \
      --moe_num_mixtures=4 \
      --deep_chain_layers=4 \
      --deep_chain_relu_cells=256 \
			--batch_size=256 \
			--run_once=True
	fi
done

