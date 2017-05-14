import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from numpy import genfromtxt


# tf.set_random_seed(777)  # for reproducibility
train_ratio = 200
my_data = genfromtxt("./data/confess_success.csv", delimiter=',')

x_train_data = my_data[:,:-1].tolist()[:train_ratio]
print(len(x_train_data))
y_train_data = my_data[:,-1].reshape(-1,1).tolist()[:train_ratio]

x_test_data = my_data[:,:-1].tolist()[train_ratio:-1]
print(len(x_test_data))
y_test_data = my_data[:,-1:].tolist()[train_ratio:-1]
print(len(y_test_data))

X = tf.placeholder(tf.float32, shape=[None, 63])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([63, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

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

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train_data, Y: y_train_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_test_data, Y: y_test_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)