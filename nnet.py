import tensorflow as tf 
import numpy as np 
import pandas as pd
import time

rng = np.random
init_time = time.time()

##############
#    DATA    #
##############
def read_data():
	##################
	# read data sets #
	##################
	train = pd.read_csv('./Data/Training_Dataset.csv')
	LBoard = pd.read_csv('./Data/Leaderboard_Dataset.csv')
	Final = pd.read_csv('./Data/Final_Dataset.csv')

	#####################
	# separate the data #
	#####################
	X_train = train.iloc[:-1, 1:-1]
	Y_train = train['actual_vote']
	Y_train = Y_train[:-1]

	lc_idx = LBoard['citizen_id']
	lboard = LBoard.iloc[1:, 1:]

	fc_idx = Final['citizen_id']
	final = Final.iloc[1:, 1:]

	######################################
	# convert the data into numpy arrays #
	######################################
	X_train = X_train.as_matrix()
	Y_train = Y_train.as_matrix()

	lc_idx = lc_idx.as_matrix()
	lboard = lboard.as_matrix()

	fc_idx = fc_idx.as_matrix()
	final = final.as_matrix()


	# citizen idx is removed from all the three data sets
	# print 'Data set details: '
	# print 'Training data: X_train: ', X_train.shape, ' and Y_train: ', Y_train.shape
	# print 'Leader board data set: ', lboard.shape
	# print 'Final data set: ', final.shape

	# X 0 to 11245
	return X_train, Y_train

def clean_data():
	##################
	# clean the data #
	##################
	X_train, Y_train = read_data()
	# convert party names into one-hot vectors 
	for idx, party in enumerate(X_train[:, 0]):
		if party == 'Centaur': # [1 0 0 0 0]
			X_train[idx, 0] = 0
		elif party == 'Odyssey': # [0 1 0 0 0]
			X_train[idx, 0] = 1
		elif party == 'Cosmos': # [0 0 1 0 0]
			X_train[idx, 0] = 2
		elif party == 'Tokugawa': # [0 0 0 1 0]
			X_train[idx, 0] = 3
		elif party == 'Ebony': # [0 0 0 0 1]
			X_train[idx, 0] = 4
		else:
			print 'No such party exists'


	# take the mean value of age
	# print X_train.shape
	# print [len(list(X_train[k, 27])) for k in range(31)]

	for idx, age in enumerate(X_train[:, 27]):
		# print age
		# print len(list(X_train[idx, 27]))
		if len(list(X_train[idx, 27])) == 5:
			a1, a2, a3, a4, a5 = list(X_train[idx, 27])
			mean_age = (int(a1)*10+int(a2) + int(a4)*10+int(a5))/2.0
		elif len(list(X_train[idx, 27])) == 3:
			a1, a2, a3 = list(X_train[idx, 27])
			mean_age = (int(a1)*10+int(a2) + 70.0)/2.0
		else:
			print 'No such format exists'
		X_train[idx, 27] = mean_age
		# print mean_age

	# delete mvar32, mvar33
	X_train = X_train[:, :-2]

	# delete mvar30
	# print X_train.shape
	# print X_train[0:10, 30]
	X_train = np.delete(X_train, 30, axis=1)
	X_train = np.delete(X_train, 0, axis=1)
	# print X_train.shape

	##########################################
	# normalize all the features b/w 0 and 1 #
	##########################################
	# print X_train[0:25, 0]
	for col in xrange(X_train.shape[1]):
		_max = X_train[:, col].max()
		_min = X_train[:, col].min()
		# if col == 0:
			# print _min
			# print _max
		X_train[:, col] /= (_max - _min)

	# print X_train[0:25, 0]

	######################################
	# convert all y into one-hot vectors #
	######################################
	replace = np.zeros((Y_train.shape[0], 5))
	for idx, row in enumerate(Y_train[:]):
		if row == 'Centaur':
			replace[idx, :] = [1, 0, 0, 0, 0]
		elif row == 'Odyssey':
			replace[idx, :] = [0, 1, 0, 0, 0]
		elif row == 'Cosmos':
			replace[idx, :] = [0, 0, 1, 0, 0]
		elif row == 'Tokugawa':
			replace[idx, :] = [0, 0, 0, 1, 0]
		elif row == 'Ebony':
			replace[idx, :] = [0, 0, 0, 0, 1]
		else:
			print 'No such party! at %dth row' % (idx, )

	Y_train = replace

	return X_train, Y_train

X_train, Y_train = clean_data()

print 'Input Data shape: ', X_train.shape
print 'Label Data shape: ', Y_train.shape

# Test the model on small dataset
X_train = X_train[:100, :]
Y_train = Y_train[:100]

X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.float32)

# details of the network
input_layer = 30
hidden_layer_1 = 30
hidden_layer_2 = 30
output_layer = 5

print '\nNetwork details...'
print 'Input size: ', input_layer
print 'Hidden layer 1 units: ', hidden_layer_1
print 'Hidden layer 2 units: ', hidden_layer_2
print 'Output layer units: ', output_layer

# graph input 
x = tf.placeholder(tf.float32, [None, input_layer])
y = tf.placeholder(tf.float32, [None, output_layer])

# model weights
print '\nInitialising random weights and biases...'
w_hidden1_vals = tf.random_normal([input_layer, hidden_layer_1])
b_hidden1_vals = tf.random_normal([hidden_layer_1])
w_hidden1 = tf.Variable(w_hidden1_vals, name='W1')
b_hidden1 = tf.Variable(b_hidden1_vals, name='B1')

w_hidden2_vals = tf.random_normal([hidden_layer_1, hidden_layer_2])
b_hidden2_vals = tf.random_normal([hidden_layer_2])
w_hidden2 = tf.Variable(w_hidden2_vals, name='W2')
b_hidden2 = tf.Variable(b_hidden2_vals, name='B2')


w_output_vals = tf.random_normal([hidden_layer_2, output_layer])
b_output_vals = tf.random_normal([output_layer])
w_output = tf.Variable(w_output_vals, name='W3')
b_output = tf.Variable(b_output_vals, name='B3')

# model
def nnet():
	# ReLU activation for the first and second hidden layer
	hidden1_activations = tf.nn.relu(tf.add(tf.matmul(x, w_hidden1), b_hidden1))
	hidden2_activations = tf.nn.relu(tf.add(tf.matmul(hidden1_activations, w_hidden2), b_hidden2))

	# linear activations for the final layer
	output_activations = tf.add(tf.matmul(hidden2_activations, w_output), b_output)

	return output_activations

pred = nnet()

# cost 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# optimizer
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# predictions
corr_pred = tf.equal(tf.argmax(y, dimension=1), tf.argmax(pred, dimension=1))
accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

init_op = tf.initialize_all_variables()

# other details
training_epochs = 1000
batch_size = 64

# launch the graph
print 'Launching the graph...'
with tf.Session() as sess:
	sess.run(init_op)

	for epoch in xrange(training_epochs):
		start = time.time()
		x_batch, y_batch = X_train, Y_train

		_, J = sess.run([optimizer, cost], feed_dict={x:X_train, y:Y_train})

		print 'Epoch: ', epoch+1, ' Cost: ', J

	print '\nDone training the model!'
	print '\nFinal cost over training set: ', sess.run(cost, feed_dict={x:X_train, y:Y_train})
	print 'Accuracy over training set: %f' % (sess.run(accuracy, feed_dict={x:X_train, y:Y_train}) * 100.0, )

	print '\nSaving the parameters...'
	np.savez('./Params/w_hidden1', w_hidden1=sess.run(w_hidden1))
	np.savez('./Params/b_hidden1', b_hidden1=sess.run(b_hidden1))
	np.savez('./Params/w_hidden2', w_hidden2=sess.run(w_hidden2))
	np.savez('./Params/b_hidden2', b_hidden2=sess.run(b_hidden2))
	np.savez('./Params/w_output', w_output=sess.run(w_output))
	np.savez('./Params/b_output', b_output=sess.run(b_output))

print '\nTotal training time: ', time.time()-init_time
