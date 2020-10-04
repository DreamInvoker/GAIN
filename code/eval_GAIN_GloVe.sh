#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# -------------------GAIN_GloVe Evaluation Shell Script--------------------

model_name=GAIN_GloVe
batch_size=32
test_batch_size=16
# binary classification threshold, automatically find optimal threshold when -1, default:-1
input_theta=${2--1}
dataset=test

nohup python3 -u test.py \
  --train_set ../data/train_annotated.json \
  --train_set_save ../data/prepro_data/train_GloVe.pkl \
  --dev_set ../data/dev.json \
  --dev_set_save ../data/prepro_data/dev_GloVe.pkl \
  --test_set ../data/${dataset}.json \
  --test_set_save ../data/prepro_data/${dataset}_GloVe.pkl \
  --use_model bilstm \
  --model_name ${model_name} \
  --pretrain_model checkpoint/GAIN_GloVe_best.pt \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --gcn_dim 512 \
  --gcn_layers 2 \
  --lstm_hidden_size 256 \
  --use_entity_type \
  --use_entity_id \
  --word_emb_size 100 \
  --finetune_word \
  --pre_train_word \
  --activation relu \
  --input_theta ${input_theta} \
  >>logs/test_${model_name}.log 2>&1 &
