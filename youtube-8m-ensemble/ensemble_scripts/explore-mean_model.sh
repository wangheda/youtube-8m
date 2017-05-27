train_path=/Youtube-8M/model_predictions/ensemble_train

for d in $(ls $train_path | sort); do
  train_data_patterns="${train_path}/${d}/*.tfrecord"
  echo "$d"
  CUDA_VISIBLE_DEVICES=0 python eval.py \
      --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
      --train_dir="../model/mean_model" \
      --model="MeanModel" \
      --echo_gap=True \
      --eval_data_patterns=$train_data_patterns | tail -n 1
  echo "------------------------------------------------"
done

for d1 in $(ls $train_path | sort); do
  for d2 in $(ls $train_path | sort); do
    if [[ "$d1" < "$d2" ]]; then
      train_data_patterns="${train_path}/${d1}/*.tfrecord,${train_path}/${d2}/*.tfrecord"
      echo "$d1,$d2"
      CUDA_VISIBLE_DEVICES=0 python eval.py \
          --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
          --train_dir="../model/mean_model" \
          --model="MeanModel" \
          --echo_gap=True \
          --eval_data_patterns=$train_data_patterns | tail -n 1
      echo "------------------------------------------------"
    fi
  done
done

for d1 in $(ls $train_path | sort); do
  for d2 in $(ls $train_path | sort); do
    if [[ "$d1" < "$d2" ]]; then
      for d3 in $(ls $train_path | sort); do
        if [[ "$d2" < "$d3" ]]; then
          train_data_patterns="${train_path}/${d1}/*.tfrecord,${train_path}/${d2}/*.tfrecord,${train_path}/${d3}/*.tfrecord"
          echo "$d1,$d2,$d3"
          CUDA_VISIBLE_DEVICES=0 python eval.py \
              --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
              --train_dir="../model/mean_model" \
              --model="MeanModel" \
              --echo_gap=True \
              --eval_data_patterns=$train_data_patterns | tail -n 1
          echo "------------------------------------------------"
        fi
      done
    fi
  done
done

for d1 in $(ls $train_path | sort); do
  for d2 in $(ls $train_path | sort); do
    if [[ "$d1" < "$d2" ]]; then
      for d3 in $(ls $train_path | sort); do
        if [[ "$d2" < "$d3" ]]; then
          for d4 in $(ls $train_path | sort); do
            if [[ "$d3" < "$d4" ]]; then
              train_data_patterns="${train_path}/${d1}/*.tfrecord,${train_path}/${d2}/*.tfrecord,${train_path}/${d3}/*.tfrecord,${train_path}/${d4}/*.tfrecord"
              echo "$d1,$d2,$d3,$d4"
              CUDA_VISIBLE_DEVICES=0 python eval.py \
                  --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
                  --train_dir="../model/mean_model" \
                  --model="MeanModel" \
                  --echo_gap=True \
                  --eval_data_patterns=$train_data_patterns | tail -n 1
              echo "------------------------------------------------"
            fi
          done
        fi
      done
    fi
  done
done

for d1 in $(ls $train_path | sort); do
  for d2 in $(ls $train_path | sort); do
    if [[ "$d1" < "$d2" ]]; then
      for d3 in $(ls $train_path | sort); do
        if [[ "$d2" < "$d3" ]]; then
          for d4 in $(ls $train_path | sort); do
            if [[ "$d3" < "$d4" ]]; then
              for d5 in $(ls $train_path | sort); do
                if [[ "$d4" < "$d5" ]]; then
                  train_data_patterns="${train_path}/${d1}/*.tfrecord,${train_path}/${d2}/*.tfrecord,${train_path}/${d3}/*.tfrecord,${train_path}/${d4}/*.tfrecord,${train_path}/${d5}/*.tfrecord"
                  echo "$d1,$d2,$d3,$d4,$d5"
                  CUDA_VISIBLE_DEVICES=0 python eval.py \
                      --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
                      --train_dir="../model/mean_model" \
                      --model="MeanModel" \
                      --echo_gap=True \
                      --eval_data_patterns=$train_data_patterns | tail -n 1
                  echo "------------------------------------------------"
                fi
              done
            fi
          done
        fi
      done
    fi
  done
done

for d1 in $(ls $train_path | sort); do
  for d2 in $(ls $train_path | sort); do
    if [[ "$d1" < "$d2" ]]; then
      for d3 in $(ls $train_path | sort); do
        if [[ "$d2" < "$d3" ]]; then
          for d4 in $(ls $train_path | sort); do
            if [[ "$d3" < "$d4" ]]; then
              for d5 in $(ls $train_path | sort); do
                if [[ "$d4" < "$d5" ]]; then
                  for d6 in $(ls $train_path | sort); do
                    if [[ "$d5" < "$d6" ]]; then
                      train_data_patterns="${train_path}/${d1}/*.tfrecord,${train_path}/${d2}/*.tfrecord,${train_path}/${d3}/*.tfrecord,${train_path}/${d4}/*.tfrecord,${train_path}/${d5}/*.tfrecord,${train_path}/${d6}/*.tfrecord"
                      echo "$d1,$d2,$d3,$d4,$d5,$d6"
                      CUDA_VISIBLE_DEVICES=0 python eval.py \
                          --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
                          --train_dir="../model/mean_model" \
                          --model="MeanModel" \
                          --echo_gap=True \
                          --eval_data_patterns=$train_data_patterns | tail -n 1
                      echo "------------------------------------------------"
                    fi
                  done
                fi
              done
            fi
          done
        fi
      done
    fi
  done
done
