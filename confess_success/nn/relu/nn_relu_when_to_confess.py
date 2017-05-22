import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
# tf.set_random_seed(777)  # for reproducibility
tf.set_random_seed(777)
keep_prob = tf.placeholder(tf.float32)

my_data = genfromtxt("../../data/confess_success_434.csv", delimiter=',')

x_data = my_data[:,:-1].tolist()
y_data = my_data[:,-1:].tolist()

x_num_of_feature = len(x_data[0])

x_train_data, x_test_data , y_train_data, y_test_data=train_test_split(x_data, y_data, test_size=0.2, random_state=20)
print(len(x_train_data))
num_of_unit = 128
X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[x_num_of_feature, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([num_of_unit]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.get_variable("W2", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([num_of_unit]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.get_variable("W3", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([num_of_unit]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.get_variable("W4", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([num_of_unit]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob)

W5 = tf.get_variable("W5", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([num_of_unit]))
# hypothesis = tf.sigmoid(tf.matmul(L4, W5) + b5)
# hypothesis = tf.sigmoid(L4, W5) + b5)
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(L5, keep_prob)

W6 = tf.get_variable("W6", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([num_of_unit]))
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L6 = tf.nn.dropout(L6, keep_prob)

W7 = tf.get_variable("W7", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([num_of_unit]))
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
L7 = tf.nn.dropout(L7, keep_prob)

W8 = tf.get_variable("W8", shape=[num_of_unit, num_of_unit],
                     initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([num_of_unit]))
L8 = tf.nn.relu(tf.matmul(L7, W6) + b6)
L8 = tf.nn.dropout(L8, keep_prob)

W9 = tf.get_variable("W9", shape=[num_of_unit, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(L8, W9) + b9)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.000005).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    accuracy_train_list = []
    accuracy_test_list = []
    step_list = []
    for step in range(100000001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train_data, Y: y_train_data, keep_prob: 0.7})

        if step % 500 == 0:
            if np.isnan(cost_val):
                break

            step_list.append(step)

            h, c, train_a = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={X: x_train_data, Y: y_train_data, keep_prob: 1})

            accuracy_train_list.append(train_a)
            h, c, a = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={X: x_test_data, Y: y_test_data, keep_prob: 1})
            # print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a, "Real (Y)", y_test_data)
            accuracy_test_list.append(a)

            print(step, cost_val, train_a, a)
            if train_a > 0.95:
                 break


    # Accuracy report
    # h, c, a = sess.run([hypothesis, predicted, accuracy],
    #                    feed_dict={X: x_test_data, Y: y_test_data})
    # print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a, "Real (Y)", y_test_data)
    #
    # h, c, a = sess.run([hypothesis, predicted, accuracy],
    #                    feed_dict={X: x_train_data, Y: y_train_data})
    # print("\nHypothesis: ", h, "\nAccuracy: ", a, "Real (Y)", y_test_data)

    print(accuracy_train_list)
    print(accuracy_test_list)

    man = 0
    woman = 0
    for data in x_test_data:
        if data[-1] == 0.0:
            woman += 1
        if data[-1] == 1.0:
            man += 1

    print("man:", man, "woman:", woman)

    gat, = plt.plot(step_list, accuracy_test_list, 'ro-')
    bat, = plt.plot(step_list, accuracy_train_list, 'bs-')
    plt.legend([gat, bat], ['test data','train data'], loc=2)
    plt.xlabel('number of step')
    plt.ylabel('accuracy')
    cell_text = []
    cell_text.append(["asd"])
    plt.table(cellText=cell_text,loc='top')
    plt.show()

