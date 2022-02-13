import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import BertPreTrainedModel,BertModel

class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, feature_vocab_size, emb_size):
        super(KeyValueMemoryNetwork, self).__init__()
        self.key_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.value_embedding = nn.Embedding(feature_vocab_size, emb_size, padding_idx=0)
        self.scale = np.power(emb_size, 0.5)

    def forward(self, key_seq, value_seq, hidden, mask_matrix):

        key_embed = self.key_embedding(key_seq)
        value_embed = self.value_embedding(value_seq)

        u = torch.bmm(hidden.float(), key_embed.transpose(1, 2))
        u = u / self.scale
        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u.float(), mask_matrix.float())
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        embedding_val = value_embed.permute(3, 0, 1, 2)
        o = torch.mul(p.float(), embedding_val.float())
        o = o.permute(1, 2, 3, 0)
        o = torch.sum(o, 2)

        aspect_len = (o != 0).sum(dim=1)
        o = o.float().sum(dim=1)
        avg_o = torch.div(o, aspect_len)
        return avg_o.type_as(hidden)


class AsaWd(BertPreTrainedModel):
    def __init__(self, config):
        super(AsaWd, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.feature_vocab_size = config.feature_vocab_size
        self.bert_dropout = config.bert_dropout

        self.bert = BertModel(config)
        self.bert_dropout = nn.Dropout(0.1)
        self.double_dense = torch.nn.Linear(self.config.hidden_size*2, config.num_labels)
        self.memory = KeyValueMemoryNetwork(
            vocab_size=config.vocab_size,
            feature_vocab_size=config.feature_vocab_size,
            emb_size=config.hidden_size,
        )
        self.memory.key_embedding = nn.Embedding.from_pretrained(self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, segment_ids, valid_ids, mem_valid_ids, key_list, dep_adj_matrix, dep_value_matrix):
        sequence_output, pooled_output = self.bert(input_ids, segment_ids)
        batch_size, max_len, feat_dim = sequence_output.shape

        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=sequence_output.device)

        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        mem_valid_ids = torch.unsqueeze(mem_valid_ids, 2)
        valid_output = valid_output * mem_valid_ids
        a = self.memory(key_list, dep_value_matrix, valid_output, dep_adj_matrix)
        pooled_output = self.bert_dropout(pooled_output)
        c = torch.cat((pooled_output, a), 1)
        logits = self.double_dense(c)

        return logits

