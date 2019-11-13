import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

def q1():
    # input
    num_input = 2
    X = tf.placeholder(shape=(None, num_input), dtype=tf.float64)

    # hidden unit
    num_hidden = 10
    w_i = tf.Variable(np.ones((num_input, num_hidden)), dtype=tf.float64)
    b_i = tf.Variable(np.ones((num_hidden)), dtype=tf.float64)
    hidden_out = tf.nn.relu(tf.add(tf.matmul(X, w_i), b_i))

    # output
    w_o = tf.Variable(np.ones((num_hidden, 1)), dtype=tf.float64)
    b_o = tf.Variable(np.ones((1)), dtype=tf.float64)
    y = tf.sigmoid(tf.add(tf.matmul(hidden_out, w_o), b_o))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([hidden_out, y], feed_dict={X: [[1, 1]]}))
        print(sess.run([hidden_out, y], feed_dict={X: [[1, -1]]}))
        print(sess.run([hidden_out, y], feed_dict={X: [[-1, -1]]}))

def q2():
    # input
    num_input = 2
    X = tf.placeholder(shape=(None, num_input), dtype=tf.float64)

    # hidden unit
    num_hidden = 10
    w_i = tf.placeholder(shape=(num_input, num_hidden), dtype=tf.float64)
    b_i = tf.placeholder(shape=(num_hidden), dtype=tf.float64)
    hidden_out = tf.nn.relu(tf.add(tf.matmul(X, w_i), b_i))

    # output
    w_o = tf.placeholder(shape=(num_hidden, 1), dtype=tf.float64)
    b_o = tf.placeholder(shape=(1), dtype=tf.float64)
    y = tf.sigmoid(tf.add(tf.matmul(hidden_out, w_o), b_o))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x1_grid = np.arange(-5, 5, 0.1)
        x2_grid = np.arange(-5, 5, 0.1)
        xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
        xx1_flatten = np.reshape(xx1, (-1, 1))
        xx2_flatten = np.reshape(xx2, (-1, 1))
        for i in range(6):
            x = np.concatenate([xx1_flatten, xx2_flatten], axis=1)
            res = sess.run(y, feed_dict={
                X: x,
                w_i: np.random.normal(size=(num_input, num_hidden)),
                b_i: np.random.normal(size=(num_hidden)),
                w_o: np.random.normal(size=(num_hidden, 1)),
                b_o: np.random.normal(size=(1))
            })
            res = np.reshape(res, xx1.shape)
            plt.contourf(x1_grid, x2_grid, res)
            plt.savefig('./screenshots/fig2-%d'%(i+1))
        print('done')



df = pd.read_csv('data/data.txt', delimiter=' ', names=['x1', 'x2', 'y2'])

q2()

print('done')