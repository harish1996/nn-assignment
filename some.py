from LinearlySeperableGenerator import base_random
from SinglePerceptron import slp
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

fig = plt.figure()
sp = fig.add_subplot(1,1,1)
line = None
weights = None

def init():
	line.set_data( [], [] )
	return line,

def animate(i):
	if i >= weights.shape[0]:
		i = weights.shape[0] - 1
	X,Y = plot_weight_lines_2d( weights[i].reshape(1,3), lim_higher= 20, lim_lower= 0 )
	line.set_data( X, Y )
	return line,


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


if __name__ == "__main__":
	# ls = [	[ 0,0,0 ],
	# 	[ 0,1,0	],
	# 	[ 1,0,0 ],
	# 	[ 0,0,1 ],
	# 	[ 0,-1,1 ],
	# 	[ -1,0,1 ]]
	# df = pd.DataFrame( ls )
	# print(df)
	df = pd.read_csv("2C2D.csv",header=None)
	network = slp.SinglePerceptron( 2, lr=0.1 )
	# print(network.W)
	# weights = np.array( [[]] )
	# weights = np.append( weights, np.append( network.bias, network.W.T ) , axis=0)
	weights = np.append(  network.bias, network.W.T  )
	weights = np.array( [weights] ) 
	# print(weights)
	# fig , ax = plt.subplots()
	

	Y = df[2]
	# Y.loc[ Y[0] == 0, 0 ] = -1
	# Y = Y.values.reshape(-1,1)
	Y = np.where( Y.values == 0, -1 , 1 ).reshape(-1,1)
	X = df.drop( 2, axis = 1 ).values


	train_x, test_x, train_y, test_y = train_test_split( X, Y, test_size = 0.2 )
	# print(Y)
	total_epochs = 100
	epochs = 0
	# ax.scatter( X[:,0], X[:,1] )
	while True:
		out = network.fit_one_epoch( train_x, train_y, bs = 5 )
		# print(out)
		epochs += 1
		# print(weights,network.W)
		# weights = np.append( weights, network.W , axis=0)
		# print(network.W.T,network.bias)
		weights = np.vstack( [ weights, np.append( network.bias, network.W.T ) ])
		# print(weights)
		
		if( ( out == train_y ).all() ):
			# print(weights)
			break
		if( epochs == total_epochs ):
			break
	# print(out == Y)
	# print(weights)
	# fig, ax = plot_weight_lines_2d( network.W, lim_higher= 20, lim_lower= 0 )
	sp.xaxis.set_ticks_position('bottom')
	sp.yaxis.set_ticks_position('left')
	sp.scatter( X[:,0], X[:,1] )

	line, = sp.plot( [], [] )

	# test = [
	# 	[ 5,20 ],
	# 	[ 4,8 ],
	# 	[15,25],
	# 	[15,40],
	# 	[10,45]
	# ]
	# test = np.array( test )
	out = network.predict( test_x )
	print((out - test_y).sum())
	print(classification_report(test_y,out))

	anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=total_epochs, interval=20)
	anim.save('basic_animation.mp4', fps=3, extra_args=['-vcodec', 'libx264'])

	plt.show()
	# plt.show()
	# print(out)
	# print(X,Y)
	# print(network)
	# print(df)