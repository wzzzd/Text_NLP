# coding: utf-8
import tensorflow as tf
from generate_text.read_utils import TextConverter
# from generate_text.model import CharRNN
from generate_text.CharRNN import CharRNN
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', './model/default/converter.pkl', 'model/default/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', './model/default/', 'checkp   oint path')
tf.flags.DEFINE_string('start_string', '我轻轻地', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 100, 'max length to generate')


def main(_):
    FLAGS.start_string = FLAGS.start_string#.decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    #获取最新的模型文件路径
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    #创建字符类
    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    #读取模型文件
    model.load(FLAGS.checkpoint_path)
    #将要输入的文本
    start = converter.text_to_arr(FLAGS.start_string)
    #输出文本
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    #转换成text，中文
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()
