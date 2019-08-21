from ..SinglePerceptron.slp import SinglePerceptronLayer 
import numpy as np
from .. import activations
from .. import loss
from ..layer import Layer

# act = slp.Signum()
# print( act( np.array([-1,-2,1,2]) ) )

# def func( *args, **kwargs ):
# 	print(args)
# 	print(kwargs)
class MultiLayerPerceptron(object):

	def __init__( self, shape, input_shape, initializer = "normal",
			activation = None, lr = None ):

		if isinstance(shape,(tuple,list)):
			self.shape = list(shape)
		else:
			raise TypeError('Could not interpret shape. shape must be a list or tuple')

		if isinstance(input_shape,int):
			if input_shape > 0:
				self.input_shape = input_shape
			else:
				raise ValueError("Input shape must be positive")
		else:
			raise TypeError("Input shape must be a positive integer")

		# Checks if the initializer is valid.
		# TODO: Delegate logic to seperate module
		if initializer and initializer in ["zeros","ones", "normal"]:
			self.initializer = initializer
		elif initializer is None:
			self.initializer = "normal"
		else:
			raise ValueError("Initializer should be one of zeros, ones, normal ") 

		activation_class = activations.get(activation)
		assert( issubclass(activation_class,activations.DifferentiableActivation) ),"Activation function must be built on the top of DifferentiableActivation class"
		
		if lr is None:
			lr = 0.1
		if isinstance( lr, float ):
			self.lr = [ lr ] * len(self.shape)
		elif isinstance( lr, (tuple,list) ):
			if len(lr) == len(self.shape):
				self.lr = lr
			else:
				raise ValueError("Learning rate should be mentioned for all layers if it is a list")
		else:
			raise TypeError("Learning rate can only be float or list/tuple") 


		self.layers = []

		# First layer is special, since the number of inputs is determined by input_shape
		self.layers.append( SinglePerceptronLayer( input_size = input_shape, nodes = self.shape[0], 
			initializer = self.initializer, lr = self.lr[0] ))
		self.layers.append( activation_class() )

		# Create Perceptron layers for all the other layers.
		for i in range(len(self.shape) - 1):
			self.layers.append( SinglePerceptronLayer( input_size = self.shape[i], nodes = self.shape[i+1],
				initializer = self.initializer, lr = self.lr[i+1] ) )
			self.layers.append( activation_class() )

	def feedforward( self, X ):

		assert( isinstance(X , np.ndarray) ),"X must be a nd array"
		assert( X.shape[1] == self.input_shape ), " Input shape doesnt match "+str(X.shape[1])+" and "+str(self.input_shape)

		self.X = X
		feed = X

		for i in range(len(self.layers)):
			feed = self.layers[i].feedforward( feed )
			# print(self.layers[i].W)
			# print(feed)
		# print(feed)
		self.out = feed

		return feed

	def backpropogate( self, Y ):

		self.loss.set_output( Y )
		# print(self.out,Y)
		# print("Loss is "+str(self.loss.feedforward(self.out)))
		# print(self.loss.feedforward())
		los = self.loss.feedforward( self.out )
		error = self.loss.backpropogate( None )

		# print(los,error)
		for i in range(len(self.layers)-1,-1,-1):
			error = self.layers[i].backpropogate( error )

		for i in range(len(self.layers)):
			self.layers[i].update()


	def _fit_one_epoch( self, X, Y, shuffle=True, bs=1 ):

		start = 0
		end = bs

		shuffle_indices = np.random.permutation(X.shape[0])
		shuffled_X = X[shuffle_indices]
		shuffled_Y = Y[shuffle_indices]

		out = np.zeros_like(Y)

		while( start < X.shape[0] ):
			batch_X = shuffled_X[start:end]
			batch_Y = shuffled_Y[start:end]

			batch_out = self.feedforward( batch_X )

			out[ shuffle_indices[start:end] ] = batch_out

			self.backpropogate( batch_Y )

			start = end
			if start == X.shape[0] - 1:
				break
			end += bs
			if end >= X.shape[0]:
				end = X.shape[0] - 1

		return out

	def fit( self, X, Y, loss_function=None, epochs=10, shuffle=True, bs=1 ):

		if loss_function is None:
			loss_function = "mse"
		loss_func = loss.get( loss_function )
		self.loss = loss_func()

		for i in range(epochs):
			out = self._fit_one_epoch( X, Y, shuffle, bs )

		return out

	def predict( self, X ):
		return self.feedforward( X )




# func( 54,"big", "excellent", big = 12, act = 54 )