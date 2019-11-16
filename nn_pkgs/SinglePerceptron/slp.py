import numpy as np
import matplotlib.pyplot as plt
from .. import activations
from ..layer import Layer
# from ...LinearlySeperableGenerator import base_random


def plot_weight_lines_2d( W, lim_lower = 0, lim_higher= 25 ):
    assert(W.shape[1]== 3),"3 and only 3 weights allowed i.e. 2 dimensions"
    fig = plt.figure()
    sp = fig.add_subplot(1,1,1)
    axis_points = np.linspace(lim_lower,lim_higher,100)
    
    sp.xaxis.set_ticks_position('bottom')
    sp.yaxis.set_ticks_position('left')
    
    for i in range(W.shape[0]):
        c = W[i,0]
        a = W[i,1]
        b = W[i,2]
        
        if b == 0:
            sp.plot(np.ones_like(axis_points)*c/a, axis_points, label=str(i))
        else:
            sp.plot(axis_points, (-a*axis_points +c) / b , label=str(i))
    plt.grid()
    plt.show()

class SinglePerceptron:
	def __init__(self, input_size, initializer = "zeros", activation = "linear", lr = 0.001 ):
		self.input_size = input_size

		self.W = np.random.normal( size= (input_size,1), scale=0.001 ) #set of 0 vectors		
		self.bias = np.random.normal( size=1 )

		activation_class = activations.get(activation)

		self.activation = activation_class()

		self.lr = lr

	def feedforward( self, X ):
		assert( isinstance(X,np.ndarray) ),"X should be np.ndarray"
		assert( X.shape[1] == self.input_size ),"Input size doesnt match"		

		before_activation = self.W.T.dot( X.T ) + self.bias # Each row corresponds to outputs to a particular input
		self.out = self.activation._apply_activation( before_activation.T )
		return self.out

	def predict( self, X ):
		return self.feedforward(X)

	def fit_one_epoch( self, X, Y, bs=1 ):
		assert( isinstance(X,np.ndarray) ),"X should be np.ndarray"
		assert( isinstance(Y,np.ndarray) ),"Y should be np.ndarray"
		assert( X.shape[0] == Y.shape[0] ),"X and Y should have same number of observations"
		assert( X.shape[1] == self.input_size ),"Input size doesnt match"

		start = 0
		end = bs
		shuffle_indices = np.random.permutation(X.shape[0])
		shuffled_X = X[shuffle_indices]
		shuffled_Y = Y[shuffle_indices]

		out = np.zeros_like( Y )

		while( start < X.shape[0] ):
			batch_X = shuffled_X[start:end]
			batch_Y = shuffled_Y[start:end]

			batch_out = self.feedforward( batch_X )
			
			out[ shuffle_indices[start:end] ] = batch_out
			
			difference = batch_Y - batch_out
			adjustable = self.lr * difference.T.dot( batch_X )

			self.W += adjustable.T
			self.bias += difference.sum()

			start = end
			if start == X.shape[0] - 1:
				break
			end += bs
			if end >= X.shape[0]:
				end = X.shape[0] - 1

		return out



class SinglePerceptronLayer(Layer):

	def __init__(self, input_size, nodes, initializer = "zeros", lr = 0.01 ):
		if isinstance(input_size,(tuple,list)):
			if len(input_size) == 1:
				input_size = input_size[0]
			else:
				raise NotImplementedError("Multidimensional Inputs not implemented yet."+
				 "Use Unwrap Layer to unwrap the output instead")
		self.input_size = input_size
		self.nodes = nodes
		self.output_shape = nodes
		self.W = np.random.normal( size= (nodes,input_size+1), scale=0.001 ) #set of 0 vectors

		self.lr = lr


	def feedforward( self,X ):
		if len(X.shape) == 1:
			X = X.reshape((1,-1))
		assert( isinstance(X,np.ndarray) ),"X should be np.ndarray"
		assert( X.shape[1] == self.input_size ),"Input size doesnt match"

		x_with_bias = np.append( np.ones( (X.shape[0],1) )*1, X, axis=1 )

		self.X = x_with_bias

		# print(self.W,x_with_bias.T)
		self.before_activation = self.W.dot( x_with_bias.T ).T # Each row corresponds to outputs to a particular input

		return self.before_activation

	def backpropogate( self, error ):
		# print("slp layer error={}".format(error))		
		self.update_weights = error.T.dot(self.X)

		self.error = error.dot( self.W[:,1:] )

		return self.error 

	def update( self ):
		self.W -= self.lr * self.update_weights

	def predict( self, X ):
		return self.feedforward(X)

	def fit_one_epoch( self, X, Y, bs=5 ):
		assert( isinstance(X,np.ndarray) ),"X should be np.ndarray"
		assert( isinstance(Y,np.ndarray) ),"Y should be np.ndarray"
		assert( X.shape[0] == Y.shape[0] ),"X and Y should have same number of observations"
		assert( X.shape[1] == self.input_size ),"Input size doesnt match"

		start = 0
		end = bs
		shuffle_indices = np.random.permutation(X.shape[0])
		shuffled_X = X[shuffle_indices]
		shuffled_Y = Y[shuffle_indices]

		x_with_bias = np.append( np.ones( (X.shape[0],1) )*1, shuffled_X, axis=1 )
		out = np.zeros_like( Y )

		while( start < X.shape[0] ):
			batch_X = shuffled_X[start:end]
			batch_Y = shuffled_Y[start:end]
			batch_X_with_bias = x_with_bias[start:end]

			batch_out = self.feedforward( batch_X )
			
			out[ shuffle_indices[start:end] ] = batch_out
			
			difference = batch_Y - batch_out
			adjustable = self.lr * difference.T.dot( batch_X_with_bias )

			self.W += adjustable
			start = end
			if start == X.shape[0] - 1:
				break
			end += bs
			if end >= X.shape[0]:
				end = X.shape[0] - 1

		return out

	def fit( self, X, Y, bs=1, epochs = 1, method = "converge" ):
		assert( isinstance(X,np.ndarray) ),"X should be np.ndarray"
		assert( isinstance(Y,np.ndarray) ),"Y should be np.ndarray"
		assert( method in [ "converge", "stop" ] ), "Method can be converge or stop only"
		assert( X.shape[0] == Y.shape[0] ),"X and Y should have same number of observations"
		assert( X.shape[1] == self.input_size ),"Input size doesnt match"
		if method == "stop":
			assert( epochs >= 1 ), "Number of epochs cannot be 0 or negative"

		while True:
			epochs -= 1
			pred_y = fit_one_epoch( X,Y, bs )
			if( (pred_y==Y).all() ):
				break
			if( epochs == 0 and method == "stop" ):
				break

		return 1

	def dummy_weights(self, W):
		self.W = W

