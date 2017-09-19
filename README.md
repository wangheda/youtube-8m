# The Monkeytyping Solution to the Youtube-8M Video Understanding Challenge

This is the solution repository of the 2nd place team monkeytyping, licensed under the Apache License 2.0.

## Dependencies

Python 2.7  
Tensorflow 1.0  
Numpy 1.12  
GNU Bash  

## Resources

For an understanding of our system, read the report of our solution: 

> https://arxiv.org/abs/1706.05150

Our source code:

> https://github.com/wangheda/youtube-8m

## Useful scripts

Training scripts (training a model may take 3-5 days) are in 

> youtube-8m-wangheda/training_scripts  
> youtube-8m-zhangteng/train_scripts  

Eval scripts for selecting best performing checkpoints

> youtube-8m-wangheda/eval_scripts  
> youtube-8m-zhangteng/eval_scripts  

Infer scripts for generating intermediate files used by ensemble scripts

> youtube-8m-wangheda/infer_scripts  
> youtube-8m-zhangteng/infer_scripts  

Ensemble scripts

> youtube-8m-ensemble/ensemble_scripts

## Paths of models and data

There are some conventions that we use in our code:

models are saved in 

> ./model

train1 data is saved in 

> /Youtube-8M/data/frame/train  
> /Youtube-8M/data/video/train  

validate1 data is saved in 

> /Youtube-8M/data/frame/validate  
> /Youtube-8M/data/video/validate  

test data is saved in 

> /Youtube-8M/data/frame/test  
> /Youtube-8M/data/video/test  

train2 data is saved in 

> /Youtube-8M/data/frame/ensemble_train  
> /Youtube-8M/data/video/ensemble_train  

validate2 data is saved in 

> /Youtube-8M/data/frame/ensemble_validate  
> /Youtube-8M/data/video/ensemble_validate  

intermediate results are stored in 

> /Youtube-8M/model_predictions/ensemble_train/[method]  
> /Youtube-8M/model_predictions/ensemble_validate/[method]  
> /Youtube-8M/model_predictions/test/[method]  

## How to generate a solution

### Single model

1. Train a single model
2. evaluate the checkpoints to get the best one
3. infer the checkpoint to get intermediate result.

### Ensemble model

1. Write a configuration file 
2. train a stacking model 
3. evaluate the stacking model and pick the best checkpoint 
4. infer the checkpoint to get a submission file 

## Note

Some of the single models are developed by Heda and some by Teng, so they are distributed in two folders. 

Bagging models are in `youtube-8m-wangheda/bagging_scripts`.

Boosting and distillation models are in `youtube-8m-wangheda/bagging_scripts`.

Cascade models are in `youtube-8m-wangheda/cascade_scripts`.

Stacking models are in `youtube-8m-ensemble/ensemble_scripts`.
