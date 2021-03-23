import json
import argparse
from os import path
import re
from tqdm import tqdm
from corenlp import StanfordCoreNLP

FULL_MODEL = './stanford-corenlp-full-2018-10-05'
punctuation = ['。', '，', '、', '：', '？', '！', '（', '）', '“', '”', '【', '】']
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']

def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char

def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line[:2] == '*#':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)
    return sentence_list, label_list

def read_txt(file_path):
    sentence_list = []
    fin = open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        raw_text = text_left + " " + aspect + " " + text_right
        sentence_list.append([lines[i].lower().strip(), lines[i+1].lower().strip(), lines[i+2].lower().strip(), raw_text])
    return sentence_list

def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word

def request_features_from_stanford(data_dir, flag):
    all_sentences = read_txt(path.join(data_dir, flag + '.raw'))
    sentences_str = []
    for sentence,aspect,polarity,raw_text in all_sentences:
        raw_text = [change(i) for i in raw_text.split(" ")]
        sentences_str.append([sentence,aspect,polarity,raw_text])
    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='en') as nlp:
        for sentence,aspect,polarity,raw_text in tqdm(sentences_str):
            props = {'timeout': '5000000','annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true' ,  'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
            results=nlp.annotate(' '.join(raw_text), properties=props)
            results["sentence"] = sentence
            results["word"] = raw_text
            results["aspect"] = aspect
            results["polarity"] = polarity
            all_data.append(results)
    assert len(all_data) == len(sentences_str)
    with open(path.join(data_dir, flag + '.stanford.json'), 'w', encoding='utf8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    args = parser.parse_args()
    request_features_from_stanford(args.data_dir, "train")
    request_features_from_stanford(args.data_dir, "test")

if __name__ == '__main__':
    main()

