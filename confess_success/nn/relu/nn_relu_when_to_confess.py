import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
# tf.set_random_seed(777)  # for reproducibility

my_data = genfromtxt("../../data/confess_success_434.csv", delimiter=',')

x_data = my_data[:,:-1].tolist()
y_data = my_data[:,-1:].tolist()

x_num_of_feature = len(x_data[0])

x_train_data, x_test_data , y_train_data, y_test_data=train_test_split(x_data,y_data,test_size=0.1,random_state=20)
print(len(x_train_data))

X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[x_num_of_feature, x_num_of_feature])
b1 = tf.Variable(tf.random_normal([x_num_of_feature]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[x_num_of_feature, x_num_of_feature])
b2 = tf.Variable(tf.random_normal([x_num_of_feature]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[x_num_of_feature, x_num_of_feature])
b3 = tf.Variable(tf.random_normal([x_num_of_feature]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[x_num_of_feature, x_num_of_feature])
b4 = tf.Variable(tf.random_normal([x_num_of_feature]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

W5 = tf.get_variable("W5", shape=[x_num_of_feature, 1])
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(L4, W5) + b5)

# W6 = tf.get_variable("W6", shape=[x_num_of_feature, x_num_of_feature])
# b6 = tf.Variable(tf.random_normal([x_num_of_feature]))
# L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
#
# W7 = tf.get_variable("W7", shape=[x_num_of_feature, x_num_of_feature])
# b7 = tf.Variable(tf.random_normal([x_num_of_feature]))
# L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
#
# W8 = tf.get_variable("W8", shape=[x_num_of_feature, 1])
# b8 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.sigmoid(tf.matmul(L7, W8) + b8)

# W9 = tf.get_variable("W9", shape=[x_num_of_feature, 1])
# b9 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.sigmoid(tf.matmul(L8, W9) + b9)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

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
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train_data, Y: y_train_data})

        if step % 100 == 0:
            step_list.append(step)
            print(step, cost_val)
            h, c, train_a = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={X: x_train_data, Y: y_train_data})
            accuracy_train_list.append(train_a)
            h, c, a = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={X: x_test_data, Y: y_test_data})
            # print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a, "Real (Y)", y_test_data)
            accuracy_test_list.append(a)

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
            woman +=1
        if data[-1] == 1.0:
            man+=1

    print("man:",man,"woman:",woman)

    gat, = plt.plot(step_list, accuracy_test_list,'ro-')
    bat, = plt.plot(step_list, accuracy_train_list, 'bs-')
    plt.legend([gat,bat],['test data','train data'],loc=2)
    plt.xlabel('number of step')
    plt.ylabel('accuracy')
    plt.show()

