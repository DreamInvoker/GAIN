import sklearn.metrics
import torch

from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.GAIN import GAIN_GloVe, GAIN_BERT
from utils import get_cuda, logging, print_params


# for ablation
# from models.GCNRE_nomention import GAIN_GloVe, GAIN_BERT


def test(model, dataloader, modelname, id2rel, input_theta=-1, output=False, is_test=False, test_prefix='dev',
         relation_num=97, ours=False):
    # ours: inter-sentence F1 in LSR

    total_recall_ignore = 0

    test_result = []
    total_recall = 0
    total_steps = len(dataloader)
    for cur_i, d in enumerate(dataloader):
        print('step: {}/{}'.format(cur_i, total_steps))

        with torch.no_grad():
            labels = d['labels']
            L_vertex = d['L_vertex']
            titles = d['titles']
            indexes = d['indexes']
            overlaps = d['overlaps']

            predictions = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],
                                distance=None,
                                entity2mention_table=d['entity2mention_table'],
                                graphs=d['graphs'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=None,
                                path_table=d['path_table'],
                                entity_graphs=d['entity_graphs'],
                                ht_pair_distance=d['ht_pair_distance']
                                )

            predict_re = torch.sigmoid(predictions)

        predict_re = predict_re.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            L = L_vertex[i]
            title = titles[i]
            index = indexes[i]
            overlap = overlaps[i]
            total_recall += len(label)

            for l in label.values():
                if not l:
                    total_recall_ignore += 1

            j = 0

            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        for r in range(1, relation_num):
                            rel_ins = (h_idx, t_idx, r)
                            intrain = label.get(rel_ins, False)

                            if (ours and (h_idx, t_idx) in overlap) or not ours:
                                test_result.append((rel_ins in label, float(predict_re[i, j, r]), intrain,
                                                    title, id2rel[r], index, h_idx, t_idx, r))

                        j += 1

    test_result.sort(key=lambda x: x[1], reverse=True)

    if ours:
        total_recall = 0
        for item in test_result:
            if item[0]:
                total_recall += 1

    pr_x = []
    pr_y = []
    correct = 0
    w = 0

    if total_recall == 0:
        total_recall = 1

    for i, item in enumerate(test_result):
        correct += item[0]
        pr_y.append(float(correct) / (i + 1))  # Precision
        pr_x.append(float(correct) / total_recall)  # Recall
        if item[1] > input_theta:
            w = i

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    theta = test_result[f1_pos][1]

    if input_theta == -1:
        w = f1_pos
        input_theta = theta

    auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
    if not is_test:
        logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
    else:
        logging(
            'ma_f1 {:3.4f} | input_theta {:3.4f} test_result P {:3.4f} test_result R {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}' \
                .format(f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))

    if output:
        # output = [x[-4:] for x in test_result[:w+1]]
        output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1],
                   'score': x[1], 'intrain': x[2],
                   'r': x[-5], 'title': x[-6]} for x in test_result[:w + 1]]
        json.dump(output, open(test_prefix + "_index.json", "w"))

    pr_x = []
    pr_y = []
    correct = correct_in_train = 0
    w = 0

    # https://github.com/thunlp/DocRED/issues/47
    for i, item in enumerate(test_result):
        correct += item[0]
        if item[0] & item[2]:
            correct_in_train += 1
        if correct_in_train == correct:
            p = 0
        else:
            p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
        pr_y.append(p)
        pr_x.append(float(correct) / total_recall)

        if item[1] > input_theta:
            w = i

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()

    auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

    logging(
        'Ignore ma_f1 {:3.4f} | inhput_theta {:3.4f} test_result P {:3.4f} test_result R {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}' \
            .format(f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))

    return f1, auc, pr_x, pr_y


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    opt.data_word_vec = word2vec

    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)
        test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',
                                    instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, batch_size=opt.test_batch_size, dataset_type='test')

        model = GAIN_BERT(opt)
    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        test_set = DGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',
                                instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, batch_size=opt.test_batch_size, dataset_type='test')

        model = GAIN_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    import gc

    del train_set
    gc.collect()

    # print(model.parameters)
    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load checkpoint from {}'.format(pretrain_model))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    model = get_cuda(model)
    model.eval()

    f1, auc, pr_x, pr_y = test(model, test_loader, model_name, id2rel=id2rel,
                               input_theta=opt.input_theta, output=True, test_prefix='test', is_test=True, ours=False)
    print('finished')
