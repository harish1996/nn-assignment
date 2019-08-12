from SinglePerceptron import slp
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

fig = plt.figure()
sp = fig.add_subplot(1,1,1)

def plot_weight_lines_2d( W, lim_lower = 0, lim_higher= 40 ):
	assert(W.shape[1]== 3),"3 and only 3 weights allowed i.e. 2 dimensions"
		
	axis_points = np.linspace(lim_lower,lim_higher,100)

	for i in range(W.shape[0]):
		c = W[i,0]
		a = W[i,1]
		b = W[i,2]

	if b == 0:
		X = np.ones_like(axis_points)*c/a
		Y = axis_points
	else:
		X = axis_points
		Y = (-a*axis_points -c) / b
	return [ X, Y ]


def plot_points2d( X, Y, subp ):

	classes = Y.shape[1]
	subp.xaxis.set_ticks_position('bottom')
	subp.yaxis.set_ticks_position('left')
	for i in range(classes):
		subp.scatter( X[Y[:,i] == 1,0], X[Y[:,i]==1,1] )

def plot_lines( W, subp ):

	assert( W.shape[1] == 3 )," Weight must have 3 columns"
	lines = W.shape[0]
	for i in range(lines):
		temp_x, temp_y = plot_weight_lines_2d( W[i].reshape(1,3), lim_higher= 75, lim_lower= 0 )
		subp.plot( temp_x, temp_y )


df = pd.read_csv( "3C2D.csv",header=None)
network = slp.SinglePerceptronLayer( 2, 3, lr=0.1 )

Y = df[ [2,3,4] ]
X = df[ [0,1] ]

Y = np.where( Y.values == 0, -1 , 1 ).reshape(-1,3)
X = X.values

train_x, test_x, train_y, test_y = train_test_split( X, Y, test_size = 0.2 )
total_epochs = 100
epochs = 0


while True:
	out = network.fit_one_epoch( train_x, train_y, bs = 5 )
	epochs += 1

	if( ( out == train_y ).all() ):
		break
	if( epochs == total_epochs ):
		break

# print(network.W)
plot_points2d( test_x, test_y, sp )
plot_lines(network.W, sp )
plt.show()
out = network.predict( test_x )
print((out - test_y).sum())
print(classification_report(test_y[:,0],out[:,0]))
print(classification_report(test_y[:,1],out[:,1]))
print(classification_report(test_y[:,2],out[:,2]))