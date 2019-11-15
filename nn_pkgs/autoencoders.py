import numpy as np 
from .. import activations
from ..layer import Layer
from SinglePerceptron.slp import SinglePerceptronLayer
from ConvolutionLayer.conv_layer import NeuralNet

class SparseLayer(SinglePerceptronLayer):
	
	def __init__(self, input_size, nodes, initializer = "zeros", lr = 0.01, sparsity=0.5, penalty=1, activation="sigmoid" ):
		self.sparsity = sparsity
		self.penalty = penalty
		super().__init__( input_size, nodes, initializer, lr )
		activation_class = activations.get(activation)
		if issubclass(activation_class,activations.DifferentialActivation):
			raise TypeError("Invalid Activation {}".format(activation))
		self.activation = activation_class( input_size )


	def feedforward(self,X):
		super().feedforward(X)
		self.after_activation = self.activation.feedforward(self.before_activation)

		return self.after_activation
		
	def backpropogate(self, error):
		error = self.activation.backpropogate( error )
		
		avg_act = np.mean(self.after_activation, axis=0)
		extra_error = self.penalty *( ( -self.sparsity / avg_act ) + ( (1-self.sparsity)/(1-avg_act) ) )
		error += extra_error

		return super().backpropogate( error )

class SparseAutoencoder(object):

	def __init__(self, input_size, hidden_size, lr=0.01, sparsity=0.5, penalty=1, activation="sigmoid"):
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.lr = lr

		self.network = NeuralNet( self.input_size )
		self.network.add( SparseLayer(input_size=self.input_size, nodes=self.hidden_size, initializer="normal",
				lr = self.lr, sparsity=sparsity, penalty=penalty, activation=activation ) )
		self.network.add( SinglePerceptronLayer( input_size=self.hidden_size, nodes=self.input_size,
				initializer="normal", lr=lr ) )
		self.network.add_loss( MSE() )

	def fit(self, X, epochs, shuffle=True, verbose=True ):
		return self.network.fit(X, X, epochs, shuffle, verbose)  

	def predict(self, X):
		return self.network.predict(X)


