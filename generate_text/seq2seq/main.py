# -*- coding: utf-8 -*-


#https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb


from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time
import tensorflow as tf
from seq2seq.util import *
from seq2seq.inference import *




# 超参数
# Number of Epochs
epochs = 1000
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


#输入是文本的index数组，输出也是文本的index数组
#index
#0:<PAD>
#1:<UNK>
#2:<GO>
#3:<EOS>

path_source_int_to_letter = "./temp/source_int_to_letter.json"
path_source_letter_to_int = "./temp/source_letter_to_int.json"
path_target_int_to_letter = "./temp/target_int_to_letter.json"
path_target_letter_to_int = "./temp/target_letter_to_int.json"

def main():


    # 读取数据
    source_data, target_data = load_data()
    # 构造映射表
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)   # 对原始数据作映射
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)   # 对label数据作映射
    # 保存映射到本地文件
    save_json(source_int_to_letter, path_source_int_to_letter)
    save_json(source_letter_to_int, path_source_letter_to_int)
    save_json(target_int_to_letter, path_target_int_to_letter)
    save_json(target_letter_to_int, path_target_letter_to_int)


    # 对字母进行转换，获取index映射
    source_data = source_data.split('\n')
    target_data = target_data.split('\n')
    source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in line] for line in source_data] # 返回[[2,323,12],[23,3,6,65],...]，不在dict里面的返回'<UNK>'
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) for letter in line] + [target_letter_to_int['<EOS>']] # 返回[[12,323,2,dict('<EOS>')],[65,6,3,23,dict('<EOS>')],...]，不在dict里面的返回'<UNK>'
                  for line in target_data]



    # 构造计算图 graph
    train_graph = tf.Graph()
    with train_graph.as_default():

        # 获得模型输入（tensor）
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

        # seq2seq的前向推导过程
        training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                           targets,
                                                                           lr,
                                                                           target_sequence_length,
                                                                           max_target_sequence_length,
                                                                           source_sequence_length,
                                                                           len(source_letter_to_int),
                                                                           len(target_letter_to_int),
                                                                           encoding_embedding_size,
                                                                           decoding_embedding_size,
                                                                           rnn_size,
                                                                           num_layers,
                                                                           target_letter_to_int,
                                                                           batch_size)
        # 反向传导过程
        # 以graph形式复制tensor变量
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
        # 用于计算损失函数时的，权重值。为了防止在计算损失时，把<PAD>也算进去，非<PAD>,即为1,<PAD>即为0
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        with tf.name_scope("optimization"):
            # Loss function,对序列logits计算加权交叉熵
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)   #计算梯度值
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]  #梯度裁剪
            train_op = optimizer.apply_gradients(capped_gradients)  #应用梯度值


    #train
    # 将数据集分割为train和validation
    # 获取训练数据
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]
    # 留出一个batch进行验证
    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                               source_letter_to_int['<PAD>'],
                               target_letter_to_int['<PAD>']))

    display_step = 50 # 每隔50轮输出loss

    checkpoint = "./model/trained_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        # 循环迭代epoch次
        for epoch_i in range(1, epochs + 1):
            # 对于每个batch，逐行输入到模型
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(get_batches(train_target, train_source, batch_size, source_letter_to_int['<PAD>'], target_letter_to_int['<PAD>'])):
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})

                if batch_i % display_step == 0:
                    # 计算validation loss
                    validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_source) // batch_size,
                                  loss,
                                  validation_loss[0]))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')



if __name__ == '__main__':
    main()
