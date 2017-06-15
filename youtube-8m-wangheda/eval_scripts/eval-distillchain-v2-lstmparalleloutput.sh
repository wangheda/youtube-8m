start=$1

GPU_ID=1
EVERY=2000
MODEL=DistillchainLstmParallelFinaloutputModel
MODEL_DIR="../model/distillchain_v2_lstmparalleloutput"

DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --train_dir="$MODEL_DIR" \
        --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
        --eval_data_pattern="/Youtube-8M/data/frame/validate/validatea*" \
        --distill_data_pattern="/Youtube-8M/model_predictions/validatea/distillation/ensemble_v2_matrix_model/*.tfrecord" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --distillation_features=False \
        --distillation_as_input=True \
        --model=$MODEL \
        --rnn_swap_memory=True \
        --lstm_cells="1024,128" \
        --moe_num_mixtures=4 \
        --batch_size=32 \
        --num_readers=1 \
        --run_once=True
	fi
done

