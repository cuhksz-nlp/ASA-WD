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

    def forward(self, key_seq, value_seq, hidden, mask_matrix,  aspect_indices):

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


class BertKVMN(nn.Module):
    def __init__(self, bert, num_labels=3, feature_vocab_size=16384, bert_dropout=0.1, incro='cat'):
        super(BertKVMN, self).__init__()
        self.config = bert.config
        self.config.num_labels = num_labels
        self.config.feature_vocab_size = feature_vocab_size
        self.config.bert_dropout = bert_dropout
        self.config.incro = incro

        self.bert = bert
        self.bert_dropout = nn.Dropout(bert_dropout)
        if self.config.incro == "cat":
            self.dense = torch.nn.Linear(self.config.hidden_size*2, num_labels)
        else:
            self.dense = torch.nn.Linear(self.config.hidden_size, num_labels)
        self.memory = KeyValueMemoryNetwork(
            vocab_size=self.config.vocab_size,
            feature_vocab_size=feature_vocab_size,
            emb_size=self.config.hidden_size,
        )
        self.memory.key_embedding = nn.Embedding.from_pretrained(self.bert.embeddings.word_embeddings.weight)


    def forward(self, inputs):

        text_bert_indices, bert_segments_ids, valid_ids, text_kv_indices,  = inputs[0], inputs[1], inputs[2], inputs[3]
        features, pos_matrix, aspect_indices = inputs[4], inputs[5], inputs[6],

        sequence_output, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        batch_size, max_len, feat_dim = sequence_output.shape

        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')

        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        aspect_indices = torch.unsqueeze(aspect_indices, 2)
        valid_output = valid_output * aspect_indices
        a = self.memory(text_kv_indices, features, valid_output,pos_matrix,  aspect_indices)
        pooled_output = self.bert_dropout(pooled_output)
        if self.config.incro == 'cat':
            c = torch.cat((pooled_output, a), 1)
            logits = self.dense(c)
        else : #sum
            c = torch.add(pooled_output, a)
            logits = self.dense(c)

        return logits

