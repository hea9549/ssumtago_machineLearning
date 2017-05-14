import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

# tf.set_random_seed(777)  # for reproducibility
train_ratio = 180

my_data = genfromtxt("../data/when_to_confess_ternary_one_hot_encoding.csv", delimiter=',')
x_data = my_data[:,:-5].tolist()
y_data = my_data[:,-5:].tolist()
x_train_data, x_test_data , y_train_data, y_test_data=train_test_split(x_data,y_data,test_size=0.05,random_state=30)

print(x_train_data)
# x_train_data = my_data[:,:-5].tolist()[:train_ratio]
# y_train_data = my_data[:,-5:].tolist()[:train_ratio]
# x_test_data = my_data[:,:-5].tolist()[train_ratio:-1]
# y_test_data = my_data[:,-5:].tolist()[train_ratio:-1]

X = tf.placeholder(tf.float32, shape=[None, 69])
Y = tf.placeholder(tf.float32, shape=[None, 5])

W = tf.Variable(tf.random_normal([69, 5]), name='weight')
b = tf.Variable(tf.random_normal([5]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)


prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(50001):
        sess.run(optimizer, feed_dict={X: x_train_data, Y: y_train_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_train_data, Y: y_train_data}))
            # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test_data}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test_data, Y: y_test_data}))

    saver = tf.train.Saver()
    saver.save(sess, '../model/softmaxclassifier.ckpt')
