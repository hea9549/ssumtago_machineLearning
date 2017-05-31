import os

from ssumPredictModel import SsumPredictModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    tf.set_random_seed(777)
    my_data = genfromtxt("./setUp/feature_man.csv", delimiter=',')
    x_data = my_data[:, :-1].tolist()
    y_data = my_data[:, -1:].tolist()

    x_num_of_feature = len(x_data[0])
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_data, y_data, test_size=0.1,
                                                                            random_state=55)

    keep_prob = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    model = SsumPredictModel(X, Y, keep_prob, unit_num=64)
    model.print_model()

    result_accuracy = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(100000001):
            cost_val, _ = sess.run([model.cost, model.train],
                                   feed_dict={X: x_train_data, Y: y_train_data, model.keep_prob: 0.6})

            if step % 100 == 0:
                c, train_a = sess.run([model.predict, model.accuracy],
                                      feed_dict={X: x_train_data, Y: y_train_data, model.keep_prob: 1})
                c, a = sess.run([model.predict, model.accuracy],
                                feed_dict={X: x_test_data, Y: y_test_data, model.keep_prob: 1})

                print("cont:", cost_val, "train accuracy:", train_a, "test accuracy:", a, "step:", step)

            if train_a > 0.9:
                break

            if np.isnan(cost_val):
                break
            else:
                result_accuracy = a

        saver = tf.train.Saver()
        saver.save(sess, './model/ssum_predict_man')

    print("final accurary:", result_accuracy)
    success = 0
    fail = 0
    for data in y_test_data:
        if data[0] == 0.0:
            fail += 1
        if data[0] == 1.0:
            success += 1

    print("success:", success, "fail:", fail)
