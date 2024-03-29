import numpy as np
import matplotlib.pyplot as plt
# from ...LinearlySeperableGenerator import base_random

class ReLU:
	def _apply_activation( self,before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = 0
		return cpy


class Signum:
	def _apply_activation( self,before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = -1
		cpy[ before_activation >= 0 ] = 1
		return cpy


def plot_weight_lines_2d( W, lim_lower = -10, lim_higher= 10 ):
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

class SinglePerceptronLayer:

	def __init__(self, input_size, nodes, initializer = "zeros", activation = "signum", lr = 0.01 ):
		self.input_size = input_size
		self.nodes = nodes

		assert( activation in [ "signum", "relu" ] ),"Activation functions can be signum or relu"

		self.W = np.zeros( (nodes,input_size+1) ) #set of 0 vectors

		if( activation == "signum" ):
			self.activation = Signum()

		else: #(activation == "relu" ):
			self.activation = ReLU()

		self.lr = lr


	def feedforward( self,X ):

		assert( isinstance(X,np.ndarray) ),"X should be np.ndarray"
		assert( X.shape[1] == self.input_size ),"Input size doesnt match"

		x_with_bias = np.append( np.ones( (X.shape[0],1) )*-1, X, axis=1 )

		before_activation = self.W.dot( x_with_bias.T ).T # Each row corresponds to outputs to a particular input
		self.out = self.activation._apply_activation( before_activation )
		return self.out

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

		x_with_bias = np.append( np.ones( (X.shape[0],1) )*-1, shuffled_X, axis=1 )
		out = np.zeros_like( Y )

		while( start < X.shape[0] ):
			batch_X = shuffled_X[start:end]
			batch_Y = shuffled_Y[start:end]
			batch_X_with_bias = x_with_bias[start:end]
			# print(batch_X.shape)
			batch_out = self.feedforward( batch_X )
			# print(batch_out.shape)
			# print(Y.shape)
			# print(shuffle_indices[start:end].shape)
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

		# print(out)
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

if __name__ == "__main__":
	print("x")