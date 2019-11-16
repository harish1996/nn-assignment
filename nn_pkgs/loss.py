import numpy as np 
from .layer import Layer 
from sklearn.metrics import log_loss

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
		try:
			ret = 0.5*( X - self.actual)**2
		except RuntimeWarning:
			print(X,self.actual)
			raise Exception("Encountered runtime warning at MSE")
		return ret

	def gradient( self ):
		return ( self.pred - self.actual )

class BCE(Loss):
	def loss_value( self, X ):
		#print(self.actual.shape,X.shape)
		return log_loss( self.actual, X.reshape(-1) )
	
	def gradient( self ):
		epsilon = 1e-15
		a = np.divide( self.actual, self.pred + epsilon )
		b = np.divide( 1 - self.actual, 1 - self.pred + epsilon )
		return -a+b

mse = mean_squared_error = MSE
bce = binary_cross_entropy = BCE

def get(identifier):
	d = globals()
	if identifier in d:
		return d[identifier]
	else:
		raise ValueError(str(identifier)+" is not a valid loss function")
