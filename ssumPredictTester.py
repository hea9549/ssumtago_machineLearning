import os

from ssumPredictModel import SsumPredictModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pymongo
import random
import time
import datetime
import sys
from statistics import mean

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")

if __name__ == "__main__":
    print(sys.argv)

    unit_num = 128
    learning_rate = 0.00007
    file_path = "./setUp/feature.csv"
    max_learning_point = 0.9
    if len(sys.argv) >= 2:
        unit_num = int(sys.argv[1])
        print("unit_num:", unit_num)

    if len(sys.argv) >= 3:
        learning_rate = float(sys.argv[2])
        print("learning_rate", learning_rate)

    if len(sys.argv) >= 4:
        file_path = str(sys.argv[3])

    if len(sys.argv) >= 5:
        max_learning_point = str(sys.argv[4])


    tf.set_random_seed(777)
    my_data = genfromtxt(file_path, delimiter=',')

    print(file_path)
    x_data = my_data[:, :-1].tolist()
    y_data = my_data[:, -1:].tolist()
    print(len(x_data))
    x_num_of_feature = len(x_data[0])

    keep_prob = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    model = SsumPredictModel(X, Y, keep_prob, unit_num, learning_rate)
    model.print_model()
    result_accuracy = 0.0
    result_array = []
    for i in range(20):
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_data, y_data, test_size=0.1,
                                                                                random_state=random.randrange(1, 200))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(100000000000):
                cost_val, _ = sess.run([model.cost, model.train],
                                       feed_dict={X: x_train_data, Y: y_train_data, model.keep_prob: 0.7})

                if step % 100 == 0:
                    c, train_a = sess.run([model.predict, model.accuracy],
                                          feed_dict={X: x_train_data, Y: y_train_data, model.keep_prob: 1})
                    c, a = sess.run([model.predict, model.accuracy],
                                    feed_dict={X: x_test_data, Y: y_test_data, model.keep_prob: 1})

                    print("cont:", cost_val, "train accuracy:", train_a, "test accuracy:", a, "step:", step)

                    if train_a > max_learning_point:
                        break

                    if np.isnan(cost_val):
                        break
                    else:
                        result_accuracy = a
                        # saver = tf.train.Saver()
                        # saver.save(sess, './model/ssum_predict_man')
            result_array.append(result_accuracy)
            sess.close()

    average_accuracy=sum(result_array) / float(len(result_array))

    db_ssumtago = client['ssumtago']
    surveyResults = db_ssumtago['surveyResults']
    surveyResult = {}
    surveyResult["surveyId"] = 1
    surveyResult["result"] = str(average_accuracy)
    surveyResult["unit_num"] = unit_num
    surveyResult["learning_rate"] = str(learning_rate)
    surveyResult["file_name"] = str(file_path)
    surveyResult["date"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    surveyResults.insert_one(surveyResult)
