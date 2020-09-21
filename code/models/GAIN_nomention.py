import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from transformers import *

from utils import get_cuda


# for no mention module ablation study

class GAIN_GloVe(nn.Module):
    def __init__(self, config):
        super(GAIN_GloVe, self).__init__()
        self.config = config

        word_emb_size = config.word_emb_size
        vocabulary_size = config.vocabulary_size
        encoder_input_size = word_emb_size
        self.activation = nn.Tanh() if config.activation == 'tanh' else nn.ReLU()

        self.word_emb = nn.Embedding(vocabulary_size, word_emb_size, padding_idx=config.word_pad)
        if config.pre_train_word:
            self.word_emb = nn.Embedding(config.data_word_vec.shape[0], word_emb_size, padding_idx=config.word_pad)
            self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec[:, :word_emb_size]))

        self.word_emb.weight.requires_grad = config.finetune_word
        if config.use_entity_type:
            encoder_input_size += config.entity_type_size
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)

        if config.use_entity_id:
            encoder_input_size += config.entity_id_size
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        self.encoder = BiLSTM(encoder_input_size, config)

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == 2 * config.lstm_hidden_size, 'gcn dim should be the lstm hidden dim * 2'
        rel_name_lists = ['intra', 'inter', 'global']
        self.GCN_layers = nn.ModuleList([dglnn.GraphConv(self.gcn_dim, self.gcn_dim, norm='right', weight=True,
                                                         bias=True, activation=self.activation)
                                         for i in range(config.gcn_layers)])

        self.bank_size = self.config.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)

        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 4 + self.gcn_dim * 5, self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_dim, edge_feat=self.gcn_dim,
                                       activation=self.activation, dropout=config.dropout)
        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)
        self.attention = Attention(self.bank_size * 2, self.gcn_dim * 4)

    def forward(self, **params):
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        '''
        src = self.word_emb(params['words'])
        mask = params['mask']
        bsz, slen, _ = src.size()

        if self.config.use_entity_type:
            src = torch.cat([src, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            src = torch.cat([src, self.entity_id_emb(params['entity_id'])], dim=-1)

        # src: [batch_size, slen, encoder_input_size]
        # src_lengths: [batchs_size]

        encoder_outputs, (output_h_t, _) = self.encoder(src, params['src_lengths'])
        encoder_outputs[mask == 0] = 0
        # encoder_outputs: [batch_size, slen, 2*encoder_hid_size]
        # output_h_t: [batch_size, 2*encoder_hid_size]

        graphs = params['graphs']

        mention_id = params['mention_id']
        features = None

        for i in range(len(graphs)):
            encoder_output = encoder_outputs[i]  # [slen, 2*encoder_hid_size]
            mention_num = torch.max(mention_id[i])
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen]
            # average word -> mention
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            x = torch.mm(select_metrix, encoder_output)  # [mention_num, 2*encoder_hid_size]

            x = torch.cat((output_h_t[i].unsqueeze(0), x), dim=0)
            # x = torch.cat((torch.max(encoder_output, dim=0)[0].unsqueeze(0), x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        # mention -> entity
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id'])
        global_info = get_cuda(torch.Tensor(bsz, self.gcn_dim))

        cur_idx = 0
        entity_graph_feature = None
        for i in range(len(graphs)):
            # average mention -> entity
            select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]
            select_metrix[0][0] = 1
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
            node_num = graphs[i].number_of_nodes('node')
            entity_representation = torch.mm(select_metrix, features[cur_idx:cur_idx + node_num])
            global_info[i] = features[cur_idx]
            cur_idx += node_num

            if entity_graph_feature is None:
                entity_graph_feature = entity_representation[1:]
            else:
                entity_graph_feature = torch.cat((entity_graph_feature, entity_representation[1:]), dim=0)

        entity_graphs = params['entity_graphs']
        entity_graph_big = dgl.batch(entity_graphs)
        output_features = [entity_graph_feature]

        for GCN_layer in self.GCN_layers:
            entity_graph_feature = GCN_layer(entity_graph_big, entity_graph_feature)
            output_features.append(entity_graph_feature)
        output_features = torch.cat(output_features, dim=-1)
        self.edge_layer(entity_graph_big, entity_graph_feature)
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        entity_graphs = dgl.unbatch(entity_graph_big)

        cur_idx = 0
        for i in range(len(entity_graphs)):
            node_num = entity_graphs[i].number_of_nodes()
            entity_bank[i, :node_num] = output_features[cur_idx:cur_idx + node_num]
            cur_idx += node_num

        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)

        # [batch_size, h_t_limit, bank_size]
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1)
        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table']
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(h_t_limit):
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break

                h = h_t_pairs[i, j, 0].item()
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue

                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].all_edges())
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0:
                    continue

                # forward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v))
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')

        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )

        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))
        return predictions


class GAIN_BERT(nn.Module):
    def __init__(self, config):
        super(GAIN_BERT, self).__init__()
        self.config = config
        self.activation = nn.Tanh() if config.activation == 'tanh' else nn.ReLU()

        if config.use_entity_type:
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)

        if config.use_entity_id:
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        self.bert = BertModel.from_pretrained(config.bert_path)
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size

        rel_name_lists = ['intra', 'inter', 'global']
        self.GCN_layers = nn.ModuleList([dglnn.GraphConv(self.gcn_dim, self.gcn_dim, norm='right', weight=True,
                                                         bias=True, activation=self.activation)
                                         for i in range(config.gcn_layers)])

        self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)

        self.dropout = nn.Dropout(self.config.dropout)

        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 4 + self.gcn_dim * 5, self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_dim, edge_feat=self.gcn_dim,
                                       activation=self.activation, dropout=config.dropout)

        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)

        self.attention = Attention(self.bank_size * 2, self.gcn_dim * 4)
        # self.attention = Attention2(self.bank_size*2, self.gcn_dim*4, self.activation, config)

    def forward(self, **params):
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        ht_pair_distance: [batch_size, h_t_limit]
        '''
        words = params['words']
        mask = params['mask']
        bsz, slen = words.size()

        encoder_outputs, sentence_cls = self.bert(input_ids=words, attention_mask=mask)
        # encoder_outputs[mask == 0] = 0

        if self.config.use_entity_type:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)

        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)
        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]

        graphs = params['graphs']

        mention_id = params['mention_id']
        features = None

        for i in range(len(graphs)):
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]
            mention_num = torch.max(mention_id[i])
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen]
            # average word -> mention
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)

            x = torch.mm(select_metrix, encoder_output)  # [mention_num, bert_hid]
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        # mention -> entity
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id'])
        global_info = get_cuda(torch.Tensor(bsz, self.gcn_dim))

        cur_idx = 0
        entity_graph_feature = None
        for i in range(len(graphs)):
            # average mention -> entity
            select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]
            select_metrix[0][0] = 1
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
            node_num = graphs[i].number_of_nodes('node')
            entity_representation = torch.mm(select_metrix, features[cur_idx:cur_idx + node_num])
            global_info[i] = features[cur_idx]
            cur_idx += node_num

            if entity_graph_feature is None:
                entity_graph_feature = entity_representation[1:]
            else:
                entity_graph_feature = torch.cat((entity_graph_feature, entity_representation[1:]), dim=0)

        entity_graphs = params['entity_graphs']
        entity_graph_big = dgl.batch(entity_graphs)
        output_features = [entity_graph_feature]
        for GCN_layer in self.GCN_layers:
            entity_graph_feature = GCN_layer(entity_graph_big, entity_graph_feature)
            output_features.append(entity_graph_feature)
        output_features = torch.cat(output_features, dim=-1)
        self.edge_layer(entity_graph_big, entity_graph_feature)
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        entity_graphs = dgl.unbatch(entity_graph_big)

        cur_idx = 0
        for i in range(len(entity_graphs)):
            node_num = entity_graphs[i].number_of_nodes()
            entity_bank[i, :node_num] = output_features[cur_idx:cur_idx + node_num]
            cur_idx += node_num

        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)

        # [batch_size, h_t_limit, bank_size]
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1)
        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table']
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(h_t_limit):
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break

                h = h_t_pairs[i, j, 0].item()
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue

                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].number_of_nodes())
                    print(entity_graphs[i].all_edges())
                    print(path_table)
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0:
                    continue

                # forward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v))
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')

        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )

        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))

        return predictions


class Attention(nn.Module):
    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        '''
        src: [src_size]
        trg: [middle_node, trg_size]
        '''

        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)

        return score.squeeze(0), value.squeeze(0)


class BiLSTM(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.lstm_hidden_size,
                            num_layers=config.nlayers, batch_first=True,
                            bidirectional=True)
        self.in_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_lengths):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                      padding_value=self.config.word_pad)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)

        src_h_t = src_h_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        src_c_t = src_c_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
        output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)

        return outputs, (output_h_t, output_c_t)


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelEdgeLayer(nn.Module):
    def __init__(self,
                 node_feat,
                 edge_feat,
                 activation,
                 dropout=0.0):
        super(RelEdgeLayer, self).__init__()
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mapping = nn.Linear(node_feat * 2, edge_feat)

    def forward(self, g, inputs):
        # g = g.local_var()

        g.ndata['h'] = inputs  # [total_mention_num, node_feat]
        g.apply_edges(lambda edges: {
            'h': self.dropout(self.activation(self.mapping(torch.cat((edges.src['h'], edges.dst['h']), dim=-1))))})
        g.ndata.pop('h')
