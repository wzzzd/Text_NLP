# -*- coding: utf-8 -*-



from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time
import tensorflow as tf
import seq2seq.util



#输入层
def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')   #target长度
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')   #最大target长度
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')   #原始数据长度

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length



#encoder
def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    '''
    构造Encoder层

    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    '''
    # Encoder embedding
    # 对序列数据执行embedding操作，输入[batch_size, sequence_length]的tensor，返回[batch_size, sequence_length, embed_dim]的tensor
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell
    # 生成两层LSTM
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    # 构建RNN，接受动态输入序列（对于不同的batch，可以接收不同的sequence_length），例如，第一个batch是[batch_size,10]，第二个batch是[batch_size,20]
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    return encoder_output, encoder_state


#decoder
#对target数据进行预处理
def process_decoder_input(data, vocab_to_int, batch_size):
        '''
        补充<GO>，并移除最后一个字符
        '''
        # cut掉最后一个字符'<EOS>'的索引
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])  #切片(data, start, end, stride)
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1) #在序列前面加入'<GO>'，tf.fill:[[1],[1],...,[1]]
        return decoder_input


#对数据进行embedding
def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,target_sequence_length, max_target_sequence_length, encoder_state, decoder_input, batch_size):
    '''
    构造Decoder层

    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''
    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))   #所有目标词的向量集合
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input) #寻找输入的所有embedding向量

    # 2. 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)]) # 构建两层的LSTM

    # 3. Output全连接层（创建全连接输出层的layer对象）
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. Training decoder，用于训练的decoder，输入是target数据
    with tf.variable_scope("decode"):
        # 得到help对象，Decoder端用来训练的函数。这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target中的真实值直接输入给RNN。
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input, sequence_length=target_sequence_length,time_major=False)
    # 构造decoder，生成基本解码器对象。output_layer代表输出层，它是一个tf.layers.Layer的对象
    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
    # 对decoder执行dynamic decoding。通过maximum_iterations参数定义最大序列长度。
    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder，用于预测的decoder，输入是decoder上一时刻的输出
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')  #shape = (batch_size,)
        # 它和TrainingHelper的区别在于它会把t-1下的输出进行embedding后再输入给RNN
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens, target_letter_to_int['<EOS>'])   #embedding,start_tokens,end_token
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper, encoder_state, output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)
    return training_decoder_output, predicting_decoder_output

#Seq2Seq
def seq2seq_model(input_data, targets,
                      lr, target_sequence_length,
                      max_target_sequence_length, source_sequence_length,
                      source_vocab_size, target_vocab_size,
                      encoder_embedding_size, decoder_embedding_size,
                      rnn_size, num_layers,
                      target_letter_to_int, batch_size):

    # encoder
    # 获取encoder的状态输出，encoder_state是N-tuple，n是LSTMCells的数量，此处为2
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoder_embedding_size)

    # decoder
    # 预处理后的decoder输入，去掉最末尾的'<EOS>'，在开始加上'<GO>.'
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    # 将状态向量与输入传递给decoder
    #返回训练的输出、预测的输出
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoder_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input,
                                                                        batch_size)
    return training_decoder_output, predicting_decoder_output


