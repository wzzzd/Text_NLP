# coding: utf-8
import tensorflow as tf
from generate_text.read_utils import TextConverter, batch_generator
from generate_text.CharRNN import CharRNN
import os
import codecs


# 以FLAGS的方式获取参数
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('name', 'default', 'name of the model')  #模型名称
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch') #batch size
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')  #time steps
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')   #隐含层数
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')   #lstm 层数
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding') #是否使用embedding 层
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding') #embedding 层的维度
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')  #步长
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')   #dropout 概率
tf.flags.DEFINE_string('input_file', './data/jay.txt', 'utf8 encoded text file')    #输入的训练文本文件
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')    #最大迭代次数
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')   #每隔多少次迭代保存一次
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')   #每多少次迭代保存一次日志
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')   #选择的最大词汇量


def main(_):
    # 创建模型文件路径
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    # 读取输入的训练文本文件
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read() #text->string #读取整个文本，并转换成一个字符

    # 创建字符转化类
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)   #由text转换成的array,shape = (71713,)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    print(converter.vocab_size)

    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
