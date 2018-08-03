# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


class CharRNN(object):
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes  #分类数
        self.num_seqs = num_seqs    #相当于batch size
        self.num_steps = num_steps  #单个序列的长度，相当于time step
        self.lstm_size = lstm_size  #hidden size
        self.num_layers = num_layers    #hidden layer
        self.learning_rate = learning_rate  #学习率，步长
        self.grad_clip = grad_clip  #梯度裁剪的上限
        self.train_keep_prob = train_keep_prob  #dropout rate
        self.use_embedding = use_embedding  #是否使用embedding
        self.embedding_size = embedding_size    #embedding size

        tf.reset_default_graph()
        self.build_inputs() #构造输入
        self.build_lstm()   #构造网络结构
        self.build_loss()   #构造损失
        self.build_optimizer()  #构造优化器
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)
            lstm_outputs = self.lstm_outputs
            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1) #将所有的输出连接在一起   #shape = (100,100,128)
            x = tf.reshape(seq_output, [-1, self.lstm_size])    #shape = (10000, 128)

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b   #shape = (10000, num_classes),num_classes是词汇量大小
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def pick_top_n(self, preds, vocab_size, top_n=5):
        p = np.squeeze(preds)   #移除长度为1的维度（轴）
        # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-top_n]] = 0   #np.argsort排序
        # 归一化概率
        p = p / np.sum(p)
        # 随机选取一个字符
        c = np.random.choice(vocab_size, 1, p=p)[0] #从68中，以概率p随机抽取1个字符
        return c




    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]    #prime 是int数组，[1,21,345,67,...]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[],shape = (68,)
        for c in prime:
            x = np.zeros((1, 1))    #[[0]]
            # 输入单个字符
            x[0, 0] = c
            print("c",c)
            print("x",x)
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
        #按照某个概率获取返回的字符
        c = self.pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
            #按照某个概率获取返回的字符
            c = self.pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
