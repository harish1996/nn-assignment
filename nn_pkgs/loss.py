import numpy as np 
from .layer import Layer 

class Loss(Layer):

	def set_output( self, Y ):
		self.actual = Y

	def gradient( self ):
		pass

	def loss_value( self, X ):
		pass

	def feedforward( self, X ):
		self.pred = X
		return self.loss_value( X )

	def backpropogate( self, error ):
		return self.gradient()

class MSE(Loss):
	def loss_value( self, X ):
		return 0.5*( X - self.actual)**2

	def gradient( self ):
		return ( self.pred - self.actual )

mse = mean_squared_error = MSE

def get(identifier):
	d = globals()
	if identifier in d:
		return d[identifier]
	else:
		raise ValueError(str(identifier)+" is not a valid loss function")