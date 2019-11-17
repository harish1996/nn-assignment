import numpy as np 
from . import activations
from .layer import Layer
from .SinglePerceptron.slp import SinglePerceptronLayer
from .networks import BatchNeuralNet
from .loss import MSE

class SparseLayer(SinglePerceptronLayer):
	
	def __init__(self, input_size, nodes, initializer = "glorot_normal", lr = 0.01, sparsity=0.5, penalty=1, activation="sigmoid" ):
		self.sparsity = sparsity
		self.penalty = penalty
		super().__init__( input_size, nodes, initializer, lr )
		activation_class = activations.get(activation)
		# print(activation_class)
		if not issubclass(activation_class,activations.DifferentiableActivation):
			raise TypeError("Invalid Activation {}".format(activation))
		self.activation = activation_class( input_size )


	def feedforward(self,X):
		super().feedforward(X)
		self.after_activation = self.activation.feedforward(self.before_activation)

		return self.after_activation
		
	def backpropogate(self, error):
		epsilon = 1e-5
		error = self.activation.backpropogate( error )
		# print("Sparse layer")
		# print(self.before_activation,self.after_activation)
		avg_act = np.mean(self.after_activation, axis=0)
		#print(avg_act)
		extra_error = self.penalty *( ( -self.sparsity / (avg_act+epsilon) ) + ( (1-self.sparsity)/(1-avg_act+epsilon) ) )
		error += extra_error
		# print(extra_error)
		return super().backpropogate( error )


class SparseAutoencoder(object):
	"""
	Sparse Autoencoder, which creates an autoencoder with the encoding layer as a sparse layer
	"""

	def __init__(self, input_size, hidden_size, lr=0.01, sparsity=0.5, penalty=1):
		"""
		Constructor for Sparse Autoencoders

		input_size	- Dimensionality of input
		hidden_size	- Required Dimensionality of the hidden encoding layer
		lr 		- Learning rate
		sparsity 	- Required sparsity
		penalty		- Penalty for sparsity constraint
		"""

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.lr = lr

		self.network = BatchNeuralNet( self.input_size )
		self.network.add( SparseLayer(input_size=(self.input_size,), nodes=self.hidden_size, initializer="glorot_normal",
				lr = self.lr, sparsity=sparsity, penalty=penalty, activation="sigmoid" ) )
		self.network.add( SinglePerceptronLayer( input_size=self.hidden_size, nodes=self.input_size,
				initializer="glorot_normal", lr=lr ) )
		self.network.add_loss( MSE() )

	def fit(self, X, epochs, shuffle=True, bs=1, verbose=True ):
		"""
		Trains the Sparse Autoencoder.

		X	- Input for training
		epochs	- Number of epochs to be trained
		shuffle	- If True, the input is shuffled during each epoch
		bs	- Batch size
		verbose	- If True, prints a progress bar while training

		returns output for the corresponding input during the last epoch
		"""
		return self.network.fit(X, X, epochs=epochs, shuffle=shuffle, bs=bs, verbose=verbose)  

	def predict(self, X):
		return self.network.predict(X)

	def encode(self, X):
		if len(X.shape) == 1:
			return self.network.feedforward_upto(X, upto=1)
		elif len(X.shape) == 2:
			encoded = np.zeros( shape=(X.shape[0], self.hidden_size) )

			for i in range(X.shape[0]):
				encoded[i] =self.network.feedforward_upto(X, upto=1)
		else:
			raise TypeError("Too many dimensions. Shape(X)= {}".format(X.shape))

	# Bug: Encode-decodes only one sample at a time
	def encode_decode(self, X):
		# out = self.network.predict(X)
		# loss = self.network.last_loss(X)
		return self.network.predict_with_loss( X, X)

	def sparse_layer(self):
		return self.network.get_layer(0)
