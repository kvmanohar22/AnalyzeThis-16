import pandas as pd
import numpy as np

rng = np.random

##################
# read data sets #
##################
train = pd.read_csv('./Data/Training_Dataset.csv')
LBoard = pd.read_csv('./Data/Leaderboard_Dataset.csv')
Final = pd.read_csv('./Data/Final_Dataset.csv')

#####################
# separate the data #
#####################
X_train = train.iloc[:, 1:-1]
Y_train = train['actual_vote']

lc_idx = LBoard['citizen_id']
lboard = LBoard.iloc[:, 1:]

fc_idx = Final['citizen_id']
final = Final.iloc[:, 1:]

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
print 'Data set details: '
print 'Training data: X_train: ', X_train.shape, ' and Y_train: ', Y_train.shape
print 'Leader board data set: ', lboard.shape
print 'Final data set: ', final.shape
