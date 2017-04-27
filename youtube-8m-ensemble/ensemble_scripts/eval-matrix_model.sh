model=$1
conf=$2
validate_path=/Youtube-8M/model_predictions/ensemble_validate

validate_data_patterns=""
for d in $(cat $conf); do
  validate_data_patterns="${validate_path}/${d}/*.tfrecord${validate_data_patterns:+,$validate_data_patterns}"
done
echo "$validate_data_patterns"

GPU_ID=0
EVERY=100
start=-1
DIR="$(pwd)"
MODEL_DIR="${DIR}/../model/${model}" \

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
        echo $checkpoint;
        if [ $checkpoint -gt $start ]; then
                echo $checkpoint;
                CUDA_VISIBLE_DEVICES=0 python eval.py \
                    --model_checkpoint_path="../model/${model}/model.ckpt-${checkpoint}" \
                    --train_dir="../model/${model}" \
                    --model="MatrixRegressionModel" \
                    --eval_data_patterns="$validate_data_patterns"
        fi
done

