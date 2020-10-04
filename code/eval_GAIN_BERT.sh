#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# binary classification threshold, automatically find optimal threshold when -1, default:-1
input_theta=${2--1}
batch_size=5
test_batch_size=16
dataset=test

# -------------------GAIN_BERT_base Evaluation Shell Script--------------------

if true; then
  model_name=GAIN_BERT_base

  nohup python3 -u test.py \
    --train_set ../data/train_annotated.json \
    --train_set_save ../data/prepro_data/train_BERT.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT.pkl \
    --test_set ../data/${dataset}.json \
    --test_set_save ../data/prepro_data/${dataset}_BERT.pkl \
    --model_name ${model_name} \
    --use_model bert \
    --pretrain_model checkpoint/GAIN_BERT_base_best.pt \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --gcn_dim 808 \
    --gcn_layers 2 \
    --bert_hid_size 768 \
    --bert_path ../PLM/bert-base-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --input_theta ${input_theta} \
    >logs/test_${model_name}.log 2>&1 &
fi

# -------------------GAIN_BERT_large Evaluation Shell Script--------------------

if false; then
  model_name=GAIN_BERT_large

  nohup python3 -u test.py \
    --train_set ../data/train_annotated.json \
    --train_set_save ../data/prepro_data/train_BERT.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT.pkl \
    --test_set ../data/${dataset}.json \
    --test_set_save ../data/prepro_data/${dataset}_BERT.pkl \
    --model_name ${model_name} \
    --use_model bert \
    --pretrain_model checkpoint/GAIN_BERT_large_best.pt \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --gcn_dim 1064 \
    --gcn_layers 2 \
    --bert_hid_size 1024 \
    --bert_path ../PLM/bert-large-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --input_theta ${input_theta} \
    >logs/test_${model_name}.log 2>&1 &
fi
