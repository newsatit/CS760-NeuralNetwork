import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import matplotlib.pyplot as plt
import neural_network as nn

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

def q3():
    # input
    num_input = 2
    X = tf.placeholder(shape=(None, num_input), dtype=tf.float64)

    # hidden unit
    num_hidden = 2
    num_hidden_layers = 5
    w_i = []
    b_i = []
    hidden_out = []
    w_i.append(tf.Variable(np.ones((num_input, num_hidden)), dtype=tf.float64))
    b_i.append(tf.Variable(np.ones((num_hidden)), dtype=tf.float64))
    hidden_out.append(tf.nn.relu(tf.add(tf.matmul(X, w_i[0]), b_i[0])))
    for i in range(1, num_hidden_layers):
        w_i.append(tf.Variable(np.ones((num_hidden, num_hidden)), dtype=tf.float64))
        b_i.append(tf.Variable(np.ones((num_hidden)), dtype=tf.float64))
        hidden_out.append(tf.nn.relu(tf.add(tf.matmul(hidden_out[-1], w_i[-1]), b_i[-1])))

    # output
    w_o = tf.Variable(np.ones((num_hidden, 1)), dtype=tf.float64)
    b_o = tf.Variable(np.ones((1)), dtype=tf.float64)
    y = tf.sigmoid(tf.add(tf.matmul(hidden_out[-1], w_o), b_o))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X_raw = [[1, 1], [1, -1], [-1, -1]]
        for x_raw in X_raw:
            print(sess.run([hidden_out, y], feed_dict={X: [x_raw]}))

def q4():

    for t in range(10):
        # input
        num_input = 2
        X = tf.placeholder(shape=(None, num_input), dtype=tf.float64)

        # hidden unit
        num_hidden = 2
        num_hidden_layers = 5
        w_i = []
        b_i = []
        hidden_out = []
        w_i.append(tf.Variable(np.random.normal(size=(num_input, num_hidden)), dtype=tf.float64))
        b_i.append(tf.Variable(np.random.normal(size=(num_hidden)), dtype=tf.float64))
        hidden_out.append(tf.nn.relu(tf.add(tf.matmul(X, w_i[0]), b_i[0])))
        for i in range(1, num_hidden_layers):
            w_i.append(tf.Variable(np.random.normal(size=(num_hidden, num_hidden)), dtype=tf.float64))
            b_i.append(tf.Variable(np.random.normal(size=(num_hidden)), dtype=tf.float64))
            hidden_out.append(tf.nn.relu(tf.add(tf.matmul(hidden_out[-1], w_i[-1]), b_i[-1])))

        # output
        w_o = tf.Variable(np.random.normal(size=(num_hidden, 1)), dtype=tf.float64)
        b_o = tf.Variable(np.random.normal(size=(1)), dtype=tf.float64)
        y = tf.sigmoid(tf.add(tf.matmul(hidden_out[-1], w_o), b_o))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x1_grid = np.arange(-5, 5, 0.1)
            x2_grid = np.arange(-5, 5, 0.1)
            xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
            xx1_flatten = np.reshape(xx1, (-1, 1))
            xx2_flatten = np.reshape(xx2, (-1, 1))

            x = np.concatenate([xx1_flatten, xx2_flatten], axis=1)
            res = sess.run(y, feed_dict={
                X: x
            })
            res = np.reshape(res, xx1.shape)
            plt.contourf(x1_grid, x2_grid, res)
            plt.savefig('./screenshots/fig4-%d'%(t+1))
            print('done')

def q2_1():
    # input
    num_input = 2
    X = tf.placeholder(shape=(None, num_input), dtype=tf.float64)

    # hidden unit
    num_hidden = 2
    w = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
    w_i = tf.Variable(
        np.array([
            [w[1], w[4]],
            [w[2], w[5]]
        ]),
        dtype=tf.float64) # shape(w_i) = (num_input, num_hidden)
    b_i = tf.Variable(np.array([w[0], w[3]]), dtype=tf.float64)
    h_u = tf.add(tf.matmul(X, w_i), b_i)
    h_v = tf.nn.relu(h_u)

    # output
    w_o = tf.Variable(np.array([[w[7]], [w[8]]]), dtype=tf.float64) # w_0.shape = (num_hidden, 1)
    b_o = tf.Variable(np.array([w[6]]), dtype=tf.float64)
    y_u = tf.add(tf.matmul(h_v, w_o), b_o)
    y_v = tf.sigmoid(y_u)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        h_u, h_v, y_u, y_v = sess.run((h_u, h_v, y_u, y_v), feed_dict={X: [[1, -1]]})
        u_a, v_a = h_u[0][0], h_v[0][0]
        u_b, v_b = h_u[0][1], h_v[0][1]
        u_c, v_c = y_u[0][0], y_v[0][0]
        print('%.5f %.5f %.5f %.5f %.5f %.5f'%(u_a, v_a, u_b, v_b, u_c, v_c))
    print('done')

def q2_2(q):
    # input, num_input = 2
    test_X = [
        np.array([[1, -1]]),
        np.array([[-0.2, 1.7]]),
        np.array([[-4, 1]]),
        np.array([[1, -1]])
    ]
    test_y = [1, 0, 0, 1]
    test_w = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        [4, 3, 2, 1, 0, -1, -2, -3, -4],
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
    ]

    for t in range(len(test_X)):
        X = test_X[t]
        # hidden unit, num_hidden = 2
        w = test_w[t]
        w_i = np.array([[w[1], w[4]], [w[2], w[5]]])
        b_i = np.array([w[0], w[3]])
        h_u = np.add(np.matmul(X, w_i), b_i)
        h_v = nn.relu(h_u)

        # output
        w_o = np.array([[w[7]], [w[8]]])
        b_o = np.array([w[6]])
        y_u = np.add(np.matmul(h_v, w_o), b_o)
        y_v = nn.sigmoid(y_u)

        # Q2.1
        u_a, v_a = h_u[0][0], h_v[0][0]
        u_b, v_b = h_u[0][1], h_v[0][1]
        u_c, v_c = y_u[0][0], y_v[0][0]
        if q == 1:
            print('Q2.1', '%.5f %.5f %.5f %.5f %.5f %.5f'%(u_a, v_a, u_b, v_b, u_c, v_c))

        #Q2.2
        y = test_y[t]
        E = 1/2*(v_c - y)**2
        partial_v_c = v_c - y
        partial_u_c = partial_v_c*v_c*(1-v_c)
        if q == 2:
            print('Q2.2', '%.5f %.5f %.5f'%(E, partial_v_c, partial_u_c))

        #Q2.3
        '''
        w_{jk} = w_i[j][k]
        w_o  (num_hidden, 1)
        '''
        # shape=(num_hidden, 1)
        partial_v_h = w_o*partial_u_c
        partial_u_h = partial_v_h*(h_u >= 0).T
        # shape=(1, num_hidden)
        if q == 3:
            print('Q2.3', '%.5f %.5f %.5f %.5f'%(partial_v_h[0][0], partial_u_h[0][0],
                                                 partial_v_h[1][0], partial_u_h[1][0]))

        #Q2.4
        # shape=(num_input, num_hidden)
        # shape=(num_hidden, num_out), h_v=
        # (num_hidden, 1), h_v(1, num_hidden) partial_u_c=
        partial_w_o = h_v.T*partial_u_c
        # (num_input, num_hidden) X(1, num_input) partial_u_h(num_hidden, 1)
        partial_w_i = np.matmul(X.T, partial_u_h.T)
        # (1, 1)
        partial_b_o = np.array([partial_u_c])
        # (1, num_hidden) partial_u_h(num_hidden, 1)
        partial_b_i = np.reshape(partial_u_h.T, b_i.shape)
        if q == 4:
            print('Q2.4', '%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f'%(
                partial_b_i[0], partial_w_i[0][0], partial_w_i[1][0],
                partial_b_i[1], partial_w_i[0][1], partial_w_i[1][1],
                partial_b_o[0], partial_w_o[0][0], partial_w_o[1][0]))

        if q == 5:
            print('Q2.5')
            # old
            print('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f'%(
                b_i[0], w_i[0][0], w_i[1][0],
                b_i[1], w_i[0][1], w_i[1][1],
                b_o[0], w_o[0][0], w_o[1][0]))
            print('%.5f'%E)
            # new
            eta = 0.1
            b_i = b_i - eta*partial_b_i
            b_o = b_o - eta*partial_b_o
            w_i = w_i - eta*partial_w_i
            w_o = w_o - eta*partial_w_o
            h_u = np.add(np.matmul(X, w_i), b_i)
            h_v = nn.relu(h_u)
            y_u = np.add(np.matmul(h_v, w_o), b_o)
            y_v = nn.sigmoid(y_u)
            u_c, v_c = y_u[0][0], y_v[0][0]
            E = 1 / 2 * (v_c - y) ** 2
            print('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f'%(
                b_i[0], w_i[0][0], w_i[1][0],
                b_i[1], w_i[0][1], w_i[1][1],
                b_o[0], w_o[0][0], w_o[1][0]))
            print('%.5f' % E)
    print('done')

def compute_training_error(w_i, b_i, w_o, b_o):
    error = 0.0
    for i in range(X_train.shape[0]):
        X = X_train[i]
        X = np.reshape(X, (1, 2))
        # hidden unit, num_hidden = 2
        h_u = np.add(np.matmul(X, w_i), b_i)
        h_v = nn.relu(h_u)

        # output
        y_u = np.add(np.matmul(h_v, w_o), b_o)
        y_v = nn.sigmoid(y_u)

        u_c, v_c = y_u[0][0], y_v[0][0]
        y = y_train[i]
        E = 1/2*(v_c - y)**2
        error += E
    return error

def q2_6():
    # input, num_input = 2
    w = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]

    w_i = np.array([[w[1], w[4]], [w[2], w[5]]])
    b_i = np.array([w[0], w[3]])
    w_o = np.array([[w[7]], [w[8]]])
    b_o = np.array([w[6]])

    errors = []
    time = []

    for round in range(0, 10001):
        i = random.randrange(0, X_train.shape[0])
        X = X_train[i]
        X = np.reshape(X, (1, 2))
        # hidden unit, num_hidden = 2
        h_u = np.add(np.matmul(X, w_i), b_i)
        h_v = nn.relu(h_u)

        # output
        y_u = np.add(np.matmul(h_v, w_o), b_o)
        y_v = nn.sigmoid(y_u)

        # Q2.1
        u_a, v_a = h_u[0][0], h_v[0][0]
        u_b, v_b = h_u[0][1], h_v[0][1]
        u_c, v_c = y_u[0][0], y_v[0][0]

        #Q2.2
        y = y_train[i]
        E = 1/2*(v_c - y)**2
        partial_v_c = v_c - y
        partial_u_c = partial_v_c*v_c*(1-v_c)

        #Q2.3
        '''
        w_{jk} = w_i[j][k]
        w_o  (num_hidden, 1)
        '''
        # shape=(num_hidden, 1)
        partial_v_h = w_o*partial_u_c
        partial_u_h = partial_v_h*(h_u >= 0).T
        # shape=(1, num_hidden)

        #Q2.4
        # shape=(num_input, num_hidden)
        # shape=(num_hidden, num_out), h_v=
        # (num_hidden, 1), h_v(1, num_hidden) partial_u_c=
        partial_w_o = h_v.T*partial_u_c
        # (num_input, num_hidden) X(1, num_input) partial_u_h(num_hidden, 1)
        partial_w_i = np.matmul(X.T, partial_u_h.T)
        # (1, 1)
        partial_b_o = np.array([partial_u_c])
        # (1, num_hidden) partial_u_h(num_hidden, 1)
        partial_b_i = np.reshape(partial_u_h.T, b_i.shape)

        eta = 0.1
        if round != 0:
            b_i = b_i - eta*partial_b_i
            b_o = b_o - eta*partial_b_o
            w_i = w_i - eta*partial_w_i
            w_o = w_o - eta*partial_w_o
            h_u = np.add(np.matmul(X, w_i), b_i)
            h_v = nn.relu(h_u)
            y_u = np.add(np.matmul(h_v, w_o), b_o)
            y_v = nn.sigmoid(y_u)
            u_c, v_c = y_u[0][0], y_v[0][0]
        # print('%.5f' % E)
        if round % 100 == 0:
            error = compute_training_error(w_i, b_i, w_o, b_o)
            errors.append(error)
            time.append(round)

    print(errors)
    print(time)
    fig, ax = plt.subplots()
    time = np.array(time)
    errors = np.array(errors)
    ax.plot(time, errors)
    ax.set(xlabel='rounds', ylabel='training set error',
           title='rounds vs training set error')
    ax.grid()
    fig.savefig('screenshots/plot.png')
    print('done')

df = pd.read_csv('data/data.txt', delimiter=' ', names=['x1', 'x2', 'y2'])

X_train = df.iloc[:, :2].to_numpy()
y_train = df.iloc[:, 2].to_numpy()

q2_6()

print('done')