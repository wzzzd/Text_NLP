# -*- coding: utf-8 -*-


#https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb




import json
import numpy as np


#数据加载
def load_data():
    with open('data/yanzi_clean.txt', 'r', encoding='utf-8') as f:
        source_data = f.read()
    with open('data/yanzi_clean_target.txt', 'r', encoding='utf-8') as f:
        target_data = f.read()
    return source_data, target_data
# def load_data():
#     with open('data/letters_source.txt', 'r', encoding='utf-8') as f:
#         source_data = f.read()
#     with open('data/letters_target.txt', 'r', encoding='utf-8') as f:
#         target_data = f.read()
#     return source_data, target_data




#预处理
def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']    #特殊字符,pad:占位符,unk:少频率词,go:,eos:终止符
    set_words = list(set([character for line in data.split('\n') for character in line]))   #获取所有的字符集合，用于构建映射词典
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int



def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]




#获取batch数据
def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):    #batch 可以被分成多少份
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列，batch之间的补全数可能都不一样
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int)) #[[20 7 5 28 28 0 0],[...],...]
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int)) #[[5 20 28 28 7 3 0 0],[...],...]

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

def save_json(dict, path):
    '''
    保存json到本地文件
    '''
    with open(path, "w") as f:
        json.dump(dict,f)


def load_json(path):
    with open(path,'r') as load_f:
         load_dict = json.load(load_f)
    return load_dict


def json_transfer_int(json, transfer_value = True):
    if transfer_value:
        result = {key:int(value) for key, value in json.items()}
    else:
        result = {int(key):value for key, value in json.items()}
    return result