import pandas as pd
import numpy as np

rng = np.random

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
	print 'Training data: X_train: ', X_train.shape, ' and Y_train: ', Y_train.shape
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
# print Y_train[0:10, :]
# print 'Final X_train shape: ', X_train.shape
# print 'Final Y_train shape: ', Y_train.shape