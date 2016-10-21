import numpy as np
rng = np.random


def clean_data(X, Y, flag=0):

	# convert party names into one-hot vectors
	one_hot_party = np.zeros((X.shape[0], 5))
	for idx, party in enumerate(X[:, 0]):
		if party == 'Centaur':
			one_hot_party[idx, :] = [1, 0, 0, 0, 0]
		elif party == 'Odyssey':
			one_hot_party[idx, :] = [0, 1, 0, 0, 0]
		elif party == 'Cosmos':
			one_hot_party[idx, :] = [0, 0, 1, 0, 0]
		elif party == 'Tokugawa':
			one_hot_party[idx, :] = [0, 0, 0, 1, 0]
		elif party == 'Ebony':
			one_hot_party[idx, :] = [0, 0, 0, 0, 1]
		else:
			print 'No such party exists'

	# combine the first 25 features and make 5 features from them
	new_feature1 = []
	new_feature2 = []
	new_feature3 = []
	new_feature4 = []
	new_feature5 = []
	for idx, var in enumerate(X[:, 1]):
		new_val = X[idx, 1] + X[idx, 6] + X[idx, 11] + X[idx, 16] + X[idx, 21]
		new_feature1.append(new_val)
		new_val = X[idx, 2] + X[idx, 7] + X[idx, 12] + X[idx, 17] + X[idx, 22]
		new_feature2.append(new_val)
		new_val = X[idx, 3] + X[idx, 8] + X[idx, 13] + X[idx, 18] + X[idx, 23]
		new_feature3.append(new_val)
		new_val = X[idx, 4] + X[idx, 9] + X[idx, 14] + X[idx, 19] + X[idx, 24]
		new_feature4.append(new_val)
		new_val = X[idx, 5] + X[idx, 10] + X[idx, 15] + X[idx, 20] + X[idx, 25]
		new_feature5.append(new_val)

	new_feature1 = np.reshape(np.array(new_feature1), (X.shape[0], 1))
	new_feature2 = np.reshape(np.array(new_feature2), (X.shape[0], 1))
	new_feature3 = np.reshape(np.array(new_feature3), (X.shape[0], 1))
	new_feature4 = np.reshape(np.array(new_feature4), (X.shape[0], 1))
	new_feature5 = np.reshape(np.array(new_feature5), (X.shape[0], 1))

	# take the mean age
	for idx, age in enumerate(X[:, 27]):
		if len(list(X[idx, 27])) == 5:
			a1, a2, a3, a4, a5 = list(X[idx, 27])
			mean_age = (int(a1)*10+int(a2) + int(a4)*10+int(a5))/2.0
		elif len(list(X[idx, 27])) == 3:
			a1, a2, a3 = list(X[idx, 27])
			mean_age = (int(a1)*10+int(a2) + 70.0)/2.0
		else:
			mean_age = 25
			print 'No such format exists'
		X[idx, 27] = mean_age

	# convert the degree into numeric value
	for idx, degree in enumerate(X[:, 30]):
		# print idx, degree
		if degree == 'Primary':
			X[idx, 30] = 1
		elif degree == 'Diploma':
			X[idx, 30] = 2
		elif degree == 'Degree':
			X[idx, 30] = 3
		elif degree == 'Masters':
			X[idx, 30] = 4

	# delete unnecesary features
	X = X[:, :-2]
	for idx in range(1, 26):
		X = np.delete(X, 1, axis=1)
	X = np.delete(X, 0, axis=1)

	# add the new features
	X = np.append(X, new_feature1, axis=1)
	X = np.append(X, new_feature2, axis=1)
	X = np.append(X, new_feature3, axis=1)
	X = np.append(X, new_feature4, axis=1)
	X = np.append(X, new_feature5, axis=1)
	X = np.append(X, one_hot_party, axis=1)

	# normalize the features
	for col in xrange(X.shape[1]):
		if not (col == 2 or col == 3 or col == 4):
			_max = X[:, col].max()
			_min = X[:, col].min()
			_mean = X[:, col].mean()

			X[:, col] = (X[:, col] - _mean)/(_max-_min)

	# convert Ys into one hot vectors
	if flag == 1:
		replace = np.zeros((Y.shape[0], 5))
		for idx, row in enumerate(Y[:]):
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
				print 'No such party! at %dth row' % (idx,)
		Y = replace
		return X, Y
	else:
		return X