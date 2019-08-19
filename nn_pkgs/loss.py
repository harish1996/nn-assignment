import numpy as np 

class Loss(object):

	def set_output( self, Y ):
		self.actual = Y

	def feedforward( self, X ):
		pass

	def backpropogate( self, X ):
		pass

class MSE(Loss):
	def feedforward( self, X ):
		self.pred = X
		return 0.5*( X - self.actual)**2

	def backpropogate( self, X ):
		return ( self.pred - self.actual )

mse = mean_squared_error = MSE

def get(identifier):
	d = globals()
	if identifier in d:
		return d[identifier]
	else:
		raise ValueError(str(identifier)+" is not a valid loss function")