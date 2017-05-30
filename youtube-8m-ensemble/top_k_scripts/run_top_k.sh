#!/bin/bash

for i in 8 12 16 20; do 
  bash top_k_scripts/train-attention_matrix_model.sh model_selection/top_${i}_model ../model/model_selection/top_${i}_model.conf 4 $(($i/4))
  bash top_k_scripts/eval-attention_matrix_model.sh model_selection/top_${i}_model ../model/model_selection/top_${i}_model.conf 4 $(($i/4))
done
