import numpy as np

class Activation(object):
	def _apply_activation( self, before_activation ):
		pass

	# Allows directly calling _apply_activation through the
	# object.
	def __call__(self, before_activation ):
		return self._apply_activation(before_activation)

class DifferentiableActivation(Activation):
	def gradient( self, before_activation ):
		pass

	def propogate_error( self, before_activation, error ):
		pass

class ReLU(DifferentiableActivation):
	def _apply_activation( self,before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = 0
		return cpy

	def gradient( self, before_activation ):
		cpy = np.array( before_activation )
		cpy[ before_activation < 0 ] = 0
		cpy[ before_activation >= 0 ] = 1
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
		return 1/( 1+np.exp(-before_activation) )

	def gradient( self, before_activation ):
		sigma = self._apply_activation( before_activation )
		return sigma*(1-sigma)

relu = ReLU
signum = Signum
linear = Linear
sigmoid = Sigmoid

def get(identifier):
	d = globals()
	if identifier in d:
		return d[identifier]
	else:
		raise ValueError(str(identifier)+" is not a valid activation function")