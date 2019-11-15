import numpy as np
from .layer import Layer
import scipy

class Activation(Layer):
	def __init__(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape
		
	def _apply_activation( self, before_activation ):
		pass

	def feedforward(self, X):
		self.before_activation = X
		return self._apply_activation(X)

	def backpropogate( self, error ):
		return np.zeros_like(error)

	# Allows directly calling _apply_activation through the
	# object.
	def __call__(self, before_activation ):
		return self._apply_activation(before_activation)

class DifferentiableActivation(Activation):
	def gradient( self, before_activation ):
		pass

	def backpropogate( self, error ):
		# print("activation layer error={}".format(error))
		return error * self.gradient( self.before_activation )

	# def propogate_error( self, before_activation, error ):
	# 	pass

class ReLU(DifferentiableActivation):
	def _apply_activation( self,before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = 0
		return cpy

	def gradient( self, before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = 0
		cpy[ before_activation >= 0 ] = 1
		# print(cpy)
		return cpy	

class Signum(Activation):
	def _apply_activation( self,before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = -1
		cpy[ before_activation >= 0 ] = 1
		return cpy

class Linear(DifferentiableActivation):
	def _apply_activation(self,before_activation):
		return before_activation

	def gradient( self, before_activation ):
		return np.ones_like(before_activation)

class Sigmoid(DifferentiableActivation):

	def _apply_activation( self, before_activation ):
		# print("before activation sigmoid = {}".format(before_activation))
		return scipy.special.expit( before_activation )
		# return 1/( 1+np.exp(-before_activation) )

	def gradient( self, before_activation ):
		sigma = self._apply_activation( before_activation )
		# print("sigmoid = {}".format(sigma))
		return sigma*(1-sigma)

class Softmax(DifferentiableActivation):
	def _apply_activation( self, before_activation ):
		# print("before activation softmax = {}".format(before_activation))
		out = scipy.special.softmax( before_activation )
		# print("after activation softmax = {}".format(out))
		return out
	
	def gradient( self, before_activation ):
		sigma = self._apply_activation( before_activation )
		# print("softmax = {}".format(sigma))
		return sigma*(1-sigma)
relu = ReLU
signum = Signum
linear = Linear
sigmoid = Sigmoid
softmax = Softmax

def get(identifier):
	d = globals()
	if identifier in d:
		return d[identifier]
	else:
		raise ValueError(str(identifier)+" is not a valid activation function")
