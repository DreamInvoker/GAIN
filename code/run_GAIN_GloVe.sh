#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# -------------------GAIN_GloVe Training Shell Script--------------------

model_name=GAIN_GloVe
lr=0.001
batch_size=32
test_batch_size=16
epoch=300
test_epoch=5
log_step=20
save_model_freq=3
negativa_alpha=4
weight_decay=0.0001

nohup python3 -u train.py \
  --train_set ../data/train_annotated.json \
  --train_set_save ../data/prepro_data/train_GloVe.pkl \
  --dev_set ../data/dev.json \
  --dev_set_save ../data/prepro_data/dev_GloVe.pkl \
  --test_set ../data/test.json \
  --test_set_save ../data/prepro_data/test_GloVe.pkl \
  --use_model bilstm \
  --model_name ${model_name} \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epoch ${epoch} \
  --test_epoch ${test_epoch} \
  --log_step ${log_step} \
  --save_model_freq ${save_model_freq} \
  --negativa_alpha ${negativa_alpha} \
  --gcn_dim 512 \
  --gcn_layers 2 \
  --lstm_hidden_size 256 \
  --use_entity_type \
  --use_entity_id \
  --word_emb_size 100 \
  --finetune_word \
  --pre_train_word \
  --dropout 0.6 \
  --activation relu \
  --weight_decay ${weight_decay} \
  >logs/train_${model_name}.log 2>&1 &

# -------------------additional options--------------------

# option below is used to resume training, it should be add into the shell scripts above
# --pretrain_model checkpoint/GAIN_GloVe_10.pt \
