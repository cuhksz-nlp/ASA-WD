import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
import json
from collections import defaultdict
from os import path
from dep_parser import DepInstanceParser

def get_features(seq_len, feature_data, feature2count, feature2id, opt):
    ret = []
    feature_text = []
    for item in feature_data:
        feature = item[opt.knowledge_type]
        word_feature = item["word"]
        if feature2count[word_feature] >= opt.f_t:
            ret.append(feature2id[word_feature])
            feature_text.append(word_feature)
        else:
            ret.append(feature2id[feature])
            feature_text.append(feature)
    ret += [0] * (seq_len - len(ret))
    return torch.tensor(ret), feature_text


def sentence2aspect(seq_len, aspect_indices,ret_sentence,pos_length):
    ret_aspect = [[0] * seq_len for _ in range(seq_len)]
    ret_whole = ret_sentence.copy()
    position = pos_length
    for index, af in enumerate(aspect_indices):
        if af == 1:
            position += 1 # [SEP] or next
            ret_aspect[position] = ret_sentence[index]
            ret_whole[position] = ret_sentence[index]
    return ret_aspect, ret_whole

def pad_graph(seq_len,graph):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(graph):
        for j in range(len(item)):
            ret[i+1][j] = item[j]
    return torch.tensor(ret)
def get_dep_mask_matrix(seq_len, feature_data, aspect_indices):
    ret = [[0] * seq_len for _ in range(seq_len)]
    det_list = []
    for i, item in enumerate(feature_data):
        dep_range = item["range"]
        for j in range(len(dep_range)):
            if dep_range[j] == 1:
                ret[i + 1][j] = 1


    ret_second = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        first_dependency = [index for index, af in enumerate(ret[i+1]) if af != 0]
        ranges = []
        for range_index in first_dependency:
            ranges.append(feature_data[range_index]["range"])
        for dep_range in ranges:
            for j in range(len(dep_range)):
                if dep_range[j] == 1:
                    if j in det_list:
                        ret_second[i + 1][j] = 1
                    else:
                        ret_second[i + 1][j] = 1
    ret_third = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(feature_data):
        second_dependency = [index for index, af in enumerate(ret_second[i+1]) if af != 0]
        ranges = []
        for range_index in second_dependency:
            ranges.append(feature_data[range_index]["range"])
        for dep_range in ranges:
            for j in range(len(dep_range)):
                if dep_range[j] == 1:
                    if j in det_list:
                        ret_third[i + 1][j] = 1
                    else:
                        ret_third[i + 1][j] = 1
    pos_length = len(feature_data) + 1
    ret_aspect,ret_whole = sentence2aspect(seq_len,aspect_indices,ret,pos_length)
    ret_aspect_second, ret_whole_second = sentence2aspect(seq_len, aspect_indices, ret_second, pos_length)
    ret_aspect_third, ret_whole_third = sentence2aspect(seq_len, aspect_indices, ret_third, pos_length)
    aspect_index = [index for index, a in enumerate(aspect_indices) if a==1]
    coverage_list = [[0] * seq_len for _ in range(3)]

    for aspect_i in aspect_index:
        for i in range(seq_len):
            if ret[aspect_i][i]!=0:
                coverage_list[0][i] = 1
            if ret_second[aspect_i][i]!=0:
                coverage_list[1][i] = 1
            if ret_third[aspect_i][i]!=0:
                coverage_list[2][i] = 1

    coverage = [coverage_list[0].count(1),coverage_list[1].count(1),coverage_list[2].count(1),len(feature_data)]
    return torch.tensor(ret),torch.tensor(ret_aspect),torch.tensor(ret_whole), \
           torch.tensor(ret_second), torch.tensor(ret_aspect_second), torch.tensor(ret_whole_second), \
           torch.tensor(ret_third),torch.tensor(ret_aspect_third),torch.tensor(ret_whole_third),\
           torch.tensor(coverage)

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def id_to_sequence(self, sequence, reverse=False, padding='post', truncating='post'):
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

class StanfordFeatureProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_json(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                data.append(json.loads(line))
        return data

    def read_feature2count(self):
        with open(path.join(self.data_dir, 'feature2count.json'), 'r', encoding='utf8') as f:
            return json.loads(f.read())

    def read_features(self, flag):
        all_data = self.read_json(path.join(self.data_dir, flag + '.spacy.json'))
        all_feature_data = []
        for data in all_data:
            tokens = []
            basicDependencies = []
            sentences=data['sentences']
            for sentence in sentences:
                tokens.extend(sentence['tokens'])
                basicDependencies.extend(sentence['basicDependencies'])

            dep_instance_parser = DepInstanceParser(basicDependencies=basicDependencies, tokens=tokens)
            first_dep_adj_matrix, first_dep_type_matrix = dep_instance_parser.get_first_order()
            second_dep_adj_matrix, second_dep_type_matrix = dep_instance_parser.get_second_order()
            third_dep_adj_matrix, third_dep_type_matrix = dep_instance_parser.get_third_order()
            all_feature_data.append({
                "words": dep_instance_parser.words,
                "first_dep_adj_matrix": first_dep_adj_matrix,
                "first_dep_type_matrix": first_dep_type_matrix,
                "second_dep_adj_matrix": second_dep_adj_matrix,
                "second_dep_type_matrix": second_dep_type_matrix,
                "third_dep_adj_matrix": third_dep_adj_matrix,
                "third_dep_type_matrix": third_dep_type_matrix
            })
        return all_feature_data

def get_feature2count(train_features, test_features, feature2type):
    feature2count = defaultdict(int)
    for sent in train_features + test_features:
        for dep_list in sent["third_dep_type_matrix"]:
            for dep_type in dep_list:
                feature2count[dep_type] += 1
    return feature2count

def generate_knowledge_api(data_dir, feature_type, flag):

    if feature_type not in ["pos", "chunk", "dep"]:
        raise RuntimeError("feature_type should be in ['pos', 'chunk', 'dep']")
    sfp = StanfordFeatureProcessor(data_dir)

    feature2count = get_feature2count(train_feature_data, test_feature_data, feature_type)

    feature2id = {"<PAD>": 0}
    id2feature = {0: "<PAD>"}
    for key in feature2count:
        index = len(feature2id)
        feature2id[key] = index
        id2feature[index] = key

    return train_feature_data, test_feature_data, feature2count, feature2id, id2feature

def get_dep_features(tokens, dep_adj_matrix, dep_type_matrix, max_seq_len, max_key_len, tokenizer, depType2id):
    final_dep_adj_matrix = [[0]*max_key_len for _ in range(max_seq_len)]
    final_dep_value_matrix = [[0]*max_key_len for _ in range(max_seq_len)]
    final_key_list = [0 for _ in range(max_key_len)]
    for i, token in enumerate(tokens):
        final_key_list[i] = tokenizer.tokenizer.convert_tokens_to_ids(tokenizer.tokenizer.tokenize(tokens[i]))[0]
    for i, token in enumerate(tokens):
        for j in range(len(dep_adj_matrix[i])):
            if j >= max_key_len:
                continue
            #because add token [CLS]
            final_dep_adj_matrix[i+1][j] = dep_adj_matrix[i][j]
            final_dep_value_matrix[i+1][j] = depType2id[dep_type_matrix[i][j]]

    return torch.tensor(final_key_list), torch.tensor(final_dep_adj_matrix), torch.tensor(final_dep_value_matrix)



class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, opt, max_key_len = 100):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        all_data = []
        print('begin generate knowledge')

        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text_right = text_right.replace("$T$", aspect)
            polarity = lines[i + 2].strip()
            raw_text = text_left + " " + aspect + " " + text_right

            # bert seg_id and bert index constructing
            text_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            raw_len = (np.sum(text_raw_indices != 0))
            aspect_bert_indices = tokenizer.text_to_sequence(aspect)
            left_bert_indices = tokenizer.text_to_sequence(text_left)
            aspect_len = np.sum(aspect_bert_indices != 0)
            bert_segments_ids = np.asarray([0] * (raw_len + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            raw_text_left_list = text_left.split()
            raw_text_right_list = text_right.split()
            raw_aspect_list = aspect.split()
            valid_indices_left, valid_indices_aspect, valid_indices_right = [], [], []
            aspect_indices_left, aspect_indices_aspect, aspect_indices_right = [], [], []


            for word in raw_text_left_list:
                token = tokenizer.tokenizer.tokenize(word)
                for m in range(len(token)):
                    if m == 0:
                        valid_indices_left.append(1)
                        aspect_indices_left.append(0)
                    else:
                        valid_indices_left.append(0)
            for word in raw_aspect_list:
                token = tokenizer.tokenizer.tokenize(word)
                for m in range(len(token)):
                    if m == 0:
                        valid_indices_aspect.append(1)
                        aspect_indices_aspect.append(1)
                    else:
                        valid_indices_aspect.append(0)
            for word in raw_text_right_list:
                token = tokenizer.tokenizer.tokenize(word)
                for m in range(len(token)):
                    if m == 0:
                        valid_indices_right.append(1)
                        aspect_indices_right.append(0)
                    else:
                        valid_indices_right.append(0)

            valid_indices = [1] + valid_indices_left + valid_indices_aspect + valid_indices_right + [1] \
                            + valid_indices_aspect + [1]
            valid_indices = tokenizer.id_to_sequence(valid_indices)

            aspect_indices = [0] + aspect_indices_left + aspect_indices_aspect + aspect_indices_right + [0]
            second_aspect_indices = [0]*len(aspect_indices)+ aspect_indices_aspect
            context_indices = [0] + [1] * (len(aspect_indices)-2)
            whole_indices = [0] + [1] * (len(aspect_indices)-2) + [0] + aspect_indices_aspect
            aspect_indices = tokenizer.id_to_sequence(aspect_indices)
            second_aspect_indices = tokenizer.id_to_sequence(second_aspect_indices)
            context_indices = tokenizer.id_to_sequence(context_indices)
            whole_indices = tokenizer.id_to_sequence(whole_indices)

            def aspect_cp_kv(dep_adj_matrix, dep_value_matrix):
                sentence_len = len(aspect_indices_left) + len(aspect_indices_aspect) + len(aspect_indices_right) + 2
                aspect_left_len = len(aspect_indices_left) + 1
                for aspect_index in range(len(aspect_indices_aspect)):
                    dep_adj_matrix[sentence_len+aspect_index] = dep_adj_matrix[aspect_left_len+aspect_index]
                    dep_value_matrix[sentence_len+aspect_index] = dep_value_matrix[aspect_left_len+aspect_index]
                return dep_adj_matrix, dep_value_matrix

            # get KV knowledge
            feature_data = feature_datas[int(i/3)]
            first_order_key_list, first_order_dep_adj_matrix, first_order_dep_value_matrix = get_dep_features(
                feature_data["words"], feature_data["first_dep_adj_matrix"], feature_data["first_dep_type_matrix"],
                tokenizer.max_seq_len, max_key_len, tokenizer, depType2id
            )
            first_order_dep_adj_matrix, first_order_dep_value_matrix = aspect_cp_kv(first_order_dep_adj_matrix, first_order_dep_value_matrix)
            second_order_key_list, second_order_dep_adj_matrix, second_order_dep_value_matrix = get_dep_features(
                feature_data["words"], feature_data["second_dep_adj_matrix"], feature_data["second_dep_type_matrix"],
                tokenizer.max_seq_len, max_key_len, tokenizer, depType2id
            )
            second_order_dep_adj_matrix, second_order_dep_value_matrix = aspect_cp_kv(second_order_dep_adj_matrix, second_order_dep_value_matrix)
            third_order_key_list, third_order_dep_adj_matrix, third_order_dep_value_matrix = get_dep_features(
                feature_data["words"], feature_data["third_dep_adj_matrix"], feature_data["third_dep_type_matrix"],
                tokenizer.max_seq_len, max_key_len, tokenizer, depType2id
            )
            third_order_dep_adj_matrix, third_order_dep_value_matrix = aspect_cp_kv(third_order_dep_adj_matrix, third_order_dep_value_matrix)

            # polarity and labels
            polarity = int(polarity) + 1

            raw_full_text = text_left + " " + aspect + " " + text_right + '   ' + aspect + '  label: ' + str(polarity)

            text_raw_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
            

            data = {
                'raw_text': raw_text,
                'aspect': aspect,
                'raw_full_text':raw_full_text,
                'text_bert_indices': text_bert_indices,
                'left_bert_indices':left_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'aspect_indices': aspect_indices,
                'second_aspect_indices': second_aspect_indices,
                'context_indices':context_indices,
                'valid_indices': valid_indices,
                'polarity': polarity,
                'text_raw_bert_indices':text_raw_bert_indices,
                'aspect_bert_indices':aspect_bert_indices,
                'whole_indices':whole_indices,
                'first_order_key_list': first_order_key_list,
                'first_order_dep_adj_matrix': first_order_dep_adj_matrix,
                'first_order_dep_value_matrix': first_order_dep_value_matrix,
                'second_order_key_list': second_order_key_list,
                'second_order_dep_adj_matrix': second_order_dep_adj_matrix,
                'second_order_dep_value_matrix': second_order_dep_value_matrix,
                'third_order_key_list': third_order_key_list,
                'third_order_dep_adj_matrix': third_order_dep_adj_matrix,
                'third_order_dep_value_matrix': third_order_dep_value_matrix
            }

            if i < 2:
                input_text = '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + ' [SEP]'
                input_text = input_text.split(" ")
                print(input_text)
                keys_list = tokenizer.tokenizer.convert_ids_to_tokens([x.item() for x in first_order_key_list if x.item() > 0])
                print(",".join(keys_list))
                for idx in range(len(input_text)):
                    print("#first_order_dep# {} value: {}".format(
                        input_text[idx],
                        ",".join(["{}-{}".format(keys_list[x],id2depType[first_order_dep_value_matrix[idx][x].item()])
                                  for x in range(first_order_dep_adj_matrix[idx].size(0))
                                  if first_order_dep_adj_matrix[idx][x].item() == 1])
                    ))

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
