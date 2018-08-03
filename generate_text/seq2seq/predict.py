# -*- coding: utf-8 -*-

__author__ = 'Administrator'


from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time
import tensorflow as tf
from seq2seq.util import *




#预测
def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001



# 读取映射字典
path_source_int_to_letter = "./temp/source_int_to_letter.json"
path_source_letter_to_int = "./temp/source_letter_to_int.json"
path_target_int_to_letter = "./temp/target_int_to_letter.json"
path_target_letter_to_int = "./temp/target_letter_to_int.json"
source_letter_to_int = load_json(path_source_letter_to_int)
source_letter_to_int = json_transfer_int(source_letter_to_int)
source_int_to_letter = load_json(path_source_int_to_letter)
source_int_to_letter = json_transfer_int(source_int_to_letter, False)
target_int_to_letter = load_json(path_target_int_to_letter)
target_int_to_letter = json_transfer_int(target_int_to_letter, False)


# 输入一个单词
generation_len = 20
input_word = '下过雨的 夏天傍晚'
text = source_to_seq(input_word)
checkpoint = "./model/trained_model.ckpt"
loaded_graph = tf.Graph()
print('原始输入:', input_word)
print('Word 编号:    {}'.format([i for i in text]))

with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    for step in range(generation_len):
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]   #[459 459 466 594 594 3]

        # 打印预测的字符
        pad = source_letter_to_int["<PAD>"]
        eos = source_letter_to_int["<EOS>"]
        print("pad:",pad)
        print("eos:",eos)
        # print('Source')
        # print('Word 编号:    {}'.format([i for i in text]))
        # print('Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))
        # print('Target')
        print('Word 编号:{}'.format([i for i in answer_logits if i != pad]))
        print('Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad and i != eos])))
        input_word = "".join([target_int_to_letter[i] for i in answer_logits if i != pad and i != eos])
        # print("answer_logits:",answer_logits)







