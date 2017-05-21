
noise_level=$1
start=$2

EVERY=1000
MODEL=LstmMemoryModel
MODEL_DIR="../model/lstmmemory1024_moe8_input_noise/noise_level_$noise_level"

DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/frame/validate/validatea*" \
			--frame_features=True \
			--feature_names="rgb,audio" \
			--feature_sizes="1024,128" \
			--model=LstmMemoryModel \
			--lstm_layers=2 \
			--lstm_cells=1024 \
			--moe_num_mixtures=8 \
			--batch_size=128 \
			--num_readers=4 \
			--run_once=True
	fi
done

