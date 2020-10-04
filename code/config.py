import argparse
import json
import os

import numpy as np

data_dir = '../data/'
prepro_dir = os.path.join(data_dir, 'prepro_data/')
if not os.path.exists(prepro_dir):
    os.mkdir(prepro_dir)

rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), "r"))
id2rel = {v: k for k, v in rel2id.items()}
word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), "r"))
ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), "r"))

word2vec = np.load(os.path.join(data_dir, 'vec.npy'))


def get_opt():
    parser = argparse.ArgumentParser()

    # datasets path
    parser.add_argument('--train_set', type=str, default=os.path.join(data_dir, 'train_annotated.json'))
    parser.add_argument('--dev_set', type=str, default=os.path.join(data_dir, 'dev.json'))
    parser.add_argument('--test_set', type=str, default=os.path.join(data_dir, 'test.json'))

    # save path of preprocessed datasets
    parser.add_argument('--train_set_save', type=str, default=os.path.join(prepro_dir, 'train.pkl'))
    parser.add_argument('--dev_set_save', type=str, default=os.path.join(prepro_dir, 'dev.pkl'))
    parser.add_argument('--test_set_save', type=str, default=os.path.join(prepro_dir, 'test.pkl'))

    # checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--fig_result_dir', type=str, default='fig_result')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')

    # task/Dataset-related
    parser.add_argument('--vocabulary_size', type=int, default=200000)
    parser.add_argument('--relation_nums', type=int, default=97)
    parser.add_argument('--entity_type_num', type=int, default=7)
    parser.add_argument('--max_entity_num', type=int, default=80)

    # padding
    parser.add_argument('--word_pad', type=int, default=0)
    parser.add_argument('--entity_type_pad', type=int, default=0)
    parser.add_argument('--entity_id_pad', type=int, default=0)

    # word embedding
    parser.add_argument('--word_emb_size', type=int, default=10)
    parser.add_argument('--pre_train_word', action='store_true')
    parser.add_argument('--data_word_vec', type=str)
    parser.add_argument('--finetune_word', action='store_true')

    # entity type embedding
    parser.add_argument('--use_entity_type', action='store_true')
    parser.add_argument('--entity_type_size', type=int, default=20)

    # entity id embedding, i.e., coreference embedding in DocRED original paper
    parser.add_argument('--use_entity_id', action='store_true')
    parser.add_argument('--entity_id_size', type=int, default=20)

    # BiLSTM
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=32)
    parser.add_argument('--lstm_dropout', type=float, default=0.1)

    # training settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--negativa_alpha', type=float, default=0.0)  # negative example nums v.s positive example num
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=1)

    # gcn
    parser.add_argument('--mention_drop', action='store_true')
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_dim', type=int, default=808)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--activation', type=str, default="relu")

    # BERT
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--bert_path', type=str, default="")
    parser.add_argument('--bert_fix', action='store_true')
    parser.add_argument('--coslr', action='store_true')
    parser.add_argument('--clip', type=float, default=-1)

    parser.add_argument('--k_fold', type=str, default="none")

    # use BiLSTM / BERT encoder, default: BiLSTM encoder
    parser.add_argument('--use_model', type=str, default="bilstm", choices=['bilstm', 'bert'],
                        help='you should choose between bert and bilstm')

    # binary classification threshold, automatically find optimal threshold when -1
    parser.add_argument('--input_theta', type=float, default=-1)

    return parser.parse_args()
