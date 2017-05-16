import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

# tf.set_random_seed(777)  # for reproducibility
train_ratio = 180

my_data = genfromtxt("../data/nn_when_to_confess.csv", delimiter=',')
x_data = my_data[:,:-5].tolist()
y_data = my_data[:,-5:].tolist()

x_num_of_feature = len(x_data[0])

x_train_data, x_test_data , y_train_data, y_test_data=train_test_split(x_data,y_data,test_size=0.2,random_state=32)
print(len(x_train_data))

X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
Y = tf.placeholder(tf.float32, shape=[None, 5])

W1 = tf.Variable(tf.random_normal([x_num_of_feature, x_num_of_feature]), name='weight1')
b1 = tf.Variable(tf.random_normal([x_num_of_feature]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([x_num_of_feature, 50]), name='weight2')
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([50, 30]), name='weight3')
b3 = tf.Variable(tf.random_normal([30]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W6 = tf.Variable(tf.random_normal([30, 5]), name='weight6')
b6 = tf.Variable(tf.random_normal([5]), name='bias6')

hypothesis = tf.nn.softmax(tf.matmul(layer3, W6) + b6)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)


prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        sess.run(optimizer, feed_dict={X: x_train_data, Y: y_train_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_train_data, Y: y_train_data}))
            # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test_data}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test_data, Y: y_test_data}))

    saver = tf.train.Saver()
    saver.save(sess, '../model/softmaxclassifier.ckpt')
