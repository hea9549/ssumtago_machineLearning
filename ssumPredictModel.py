import tensorflow as tf


class SsumPredictModel:
    def __init__(self, input_data, label, keep_prob, unit_num=128, learning_rate=0.00007):
        self.input_data = input_data
        self.label = label
        self.unit_num = unit_num
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.hypothesis = self.hypothesis()
        self.cost = self.cost()
        self.train = self.train()
        self.predict = self.predict()
        self.accuracy = self.accuracy()

    def print_model(self):
        print("shape:",self.input_data.shape)
        print("number of unit:", self.unit_num)
        print("learning_rate", self.learning_rate)

    def hypothesis(self):
        W1 = tf.get_variable("W1", shape=[self.input_data.shape[1], self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([self.unit_num]))
        L1 = tf.nn.relu(tf.matmul(self.input_data, W1) + b1)
        L1 = tf.nn.dropout(L1, self.keep_prob)

        W2 = tf.get_variable("W2", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([self.unit_num]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, self.keep_prob)

        W3 = tf.get_variable("W3", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([self.unit_num]))
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
        L3 = tf.nn.dropout(L3, self.keep_prob)

        W4 = tf.get_variable("W4", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([self.unit_num]))
        L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
        L4 = tf.nn.dropout(L4, self.keep_prob)

        W5 = tf.get_variable("W5", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([self.unit_num]))
        L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
        L5 = tf.nn.dropout(L5, self.keep_prob)

        W6 = tf.get_variable("W6", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b6 = tf.Variable(tf.random_normal([self.unit_num]))
        L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
        L6 = tf.nn.dropout(L6, self.keep_prob)

        W7 = tf.get_variable("W7", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b7 = tf.Variable(tf.random_normal([self.unit_num]))
        L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
        L7 = tf.nn.dropout(L7, self.keep_prob)

        W8 = tf.get_variable("W8", shape=[self.unit_num, self.unit_num],
                             initializer=tf.contrib.layers.xavier_initializer())
        b8 = tf.Variable(tf.random_normal([self.unit_num]))
        L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
        L8 = tf.nn.dropout(L8, self.keep_prob)

        W9 = tf.get_variable("W9", shape=[self.unit_num, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b9 = tf.Variable(tf.random_normal([1]))
        hypothesis = tf.sigmoid(tf.matmul(L8, W9) + b9)

        return hypothesis

    def cost(self):
        cost = -tf.reduce_mean(self.label * tf.log(self.hypothesis) + (1 - self.label) *
                               tf.log(1 - self.hypothesis))
        return cost

    def train(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def predict(self):
        return tf.cast(self.hypothesis > 0.5, dtype=tf.float32)

    def accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(self.predict, self.label), dtype=tf.float32))
