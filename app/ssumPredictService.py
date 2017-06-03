import tensorflow as tf
from ssumPredictModel import SsumPredictModel

x_num_of_feature = 82
num_of_unit = 256

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

model = SsumPredictModel(X, Y, keep_prob)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
new_saver = tf.train.import_meta_graph('./model/ssum_predict_man.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./model'))


def predict_female(data):
    pass


def predict_male(data):
    pass


def predict_human(data):
    predict, hypothesis = sess.run([model.predict, model.hypothesis], feed_dict={X: data, keep_prob: 1})
    print(predict)
    print(hypothesis)
    return predict

# if __name__ == '__main__':
