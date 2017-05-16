import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
# tf.set_random_seed(777)  # for reproducibility

my_data = genfromtxt("../data/confess_success_434.csv", delimiter=',')

x_data = my_data[:,:-1].tolist()
y_data = my_data[:,-1:].tolist()

x_num_of_feature = len(x_data[0])

x_train_data, x_test_data , y_train_data, y_test_data=train_test_split(x_data,y_data,test_size=0.15,random_state=40)
print(len(x_train_data))

X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([x_num_of_feature, x_num_of_feature*2]), name='weight1')
b1 = tf.Variable(tf.random_normal([x_num_of_feature*2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([x_num_of_feature*2, x_num_of_feature*2]), name='weight2')
b2 = tf.Variable(tf.random_normal([x_num_of_feature*2]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([x_num_of_feature*2, x_num_of_feature]), name='weight3')
b3 = tf.Variable(tf.random_normal([x_num_of_feature]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([x_num_of_feature, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# W5 = tf.Variable(tf.random_normal([x_num_of_feature*2, x_num_of_feature]), name='weight5')
# b5 = tf.Variable(tf.random_normal([x_num_of_feature]), name='bias5')
# layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)
#
# W6 = tf.Variable(tf.random_normal([x_num_of_feature, 1]), name='weight6')
# b6 = tf.Variable(tf.random_normal([1]), name='bias6')
# hypothesis = tf.sigmoid(tf.matmul(layer5, W6) + b6)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    accuracy_train_list = []
    accuracy_test_list = []
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train_data, Y: y_train_data})

        if step % 200 == 0:

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
    gat,=plt.plot(accuracy_train_list,accuracy_test_list,'-ro')
    plt.legend([gat],['asd'],loc=1)
    plt.show()