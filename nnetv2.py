from log_reg import clean_data
import pandas as pd
import tensorflow as tf
import numpy as np
import time

rng = np.random

train = pd.read_csv('./Data/Training_Dataset.csv')
L_Board = pd.read_csv('./Data/Leaderboard_Dataset.csv')

X_train = train.iloc[:-1, 1:-1]
Y_train = train['actual_vote']
Y_train = Y_train[:-1]

citizen_idx = L_Board['citizen_id']
citizen_idx = citizen_idx[:-1]
l_data = L_Board.iloc[:-1, 1:]

# convert into numpy arrays
X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()
citizen_idx = citizen_idx.as_matrix()
l_data = l_data.as_matrix()

print 'Initial training data: ', X_train.shape
print 'Initial leader board data: ', l_data.shape
X_train, Y_train = clean_data(X_train, Y_train, 1)
l_data = clean_data(l_data, Y_train, 0)
print '\nFinal training data: ', X_train.shape
print 'Final leader board data: ', l_data.shape

# print X_train[0:20, :]
# time.sleep(100)
X_train = X_train[0:100, :]
Y_train = Y_train[0:100]

print 'Implementing logistic regression...'
input_layer = 16
output_layer = 5
print 'Input size: ', input_layer
print 'Output classes: ', output_layer

# input for the graph
x = tf.placeholder(tf.float32, [None, input_layer])
y = tf.placeholder(tf.float32, [None, output_layer])

# weight and biases
W_vals = np.asarray(rng.randn(input_layer, output_layer)).astype(np.float32)
W = tf.Variable(W_vals, name='W')
b_vals = np.asarray(rng.randn(output_layer)).astype(np.float32)
b = tf.Variable(b_vals, name='b')

# model
pred = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

# cost
cost = tf.reduce_mean(tf.reduce_sum(tf.pow((pred-y), 2)))

# optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# predictions
corr_pred = tf.equal(tf.argmax(y, dimension=1), tf.argmax(pred, dimension=1))
accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

init_op = tf.initialize_all_variables()

# other details
training_epochs = 1000

print '\nLaunching the graph...'
with tf.Session() as sess:
	sess.run(init_op)

	for epoch in xrange(training_epochs):
		sess.run(optimizer, feed_dict={x:X_train, y:Y_train})
		J = sess.run(cost, feed_dict={x:X_train, y:Y_train})

		print 'Epoch: ', epoch+1, ' Cost: ', J
		# _p = sess.run(pred, feed_dict={x:X_train, y:Y_train})
		# print 'pred: \n', _p
		# print 'log: \n', sess.run(tf.log(_p))
		#
		# print 'cost: ', sess.run(cost, feed_dict={x:X_train, y:Y_train})
		# sess.run(optimizer, feed_dict={x:X_train, y:Y_train})
		# # time.sleep(1)
	print '\nAccuracy over training set: ', sess.run(accuracy, feed_dict={x:X_train, y:Y_train}) * 100












