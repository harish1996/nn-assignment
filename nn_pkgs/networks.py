import numpy as np


class NeuralNet:
	def __init__(self, input_shape ):
		self.input_shape = input_shape
		self.layers = []

	def add( self, layer ):
		if isinstance(layer,Layer):
			self.layers.append(layer)

	def add_loss(self, loss ):
		if isinstance(loss,Loss):
			self.loss = loss

	def get_layer(self, layer):
		return self.layers[layer]

	def __check_shape( actual_shape ):
		if len(actual_shape) == 1 and isinstance(self.input_shape,int):
			self.input_shape = ( self.input_shape ,)
			# print(self.input_shape)
		if actual_shape != self.input_shape:
			raise TypeError("Input shape {} is not matching with configured shape {}".format(actual_shape,self.input_shape))

	def feedforward(self,X):
		self.out = self.feedforward_upto( X )
		return self.out

	def feedforward_upto(self, X, upto= None ):

		self.__check_shape( X.shape )

		if upto is None:
			upto = len(self.layers)
		if upto < 0 or upto > len(self.layers):
			raise ValueError("Layer no. {} doesn't exist".format(layer))

		feed = X

		for i in range(upto):
			feed = self.layers[i].feedforward( feed )

		return feed

	def backpropogate( self, Y ):
		self.loss.set_output(Y)

		loss = self.loss.feedforward(self.out)
		# print("loss={}".format(loss))
		error = self.loss.backpropogate(None)

		for i in range(len(self.layers)-1,-1,-1):
			error = self.layers[i].backpropogate( error )

		for i in range(len(self.layers)):
			self.layers[i].update()
		return loss

	def _fit_one_epoch( self, X, Y, shuffle=True, verbose= True ):

		shuffle_indices = np.random.permutation(X.shape[0])
		out = np.zeros_like(Y)
		loss = 0
		# print(Y)
		if verbose:
			pbar = tqdm.tqdm( total=shuffle_indices.shape[0], unit="examples" )

		for i in range(shuffle_indices.shape[0]):
			sample_out = self.feedforward( X[i] )
			# print(sample_out)
			# print(out.shape)
			out[i] = sample_out
			loss += self.backpropogate( Y[i] )
			if verbose:
				pbar.update(1)
		print("Loss is {}".format(np.sum(loss)))
		return out

	def fit( self, X, Y, epochs, shuffle=True, verbose=True ):

		if verbose:
			pbar = tqdm.tqdm( total=epochs, unit="epochs" )

		for i in range(epochs):
			out = self._fit_one_epoch( X,Y, shuffle, verbose )
			if verbose:
				pbar.update(1)

		return out

	def _predict_one_sample( self, X ):
		return self.feedforward(X)

	def predict( self, X, verbose=True ):

		if verbose:
			pbar = tqdm.tqdm( total=X.shape[0], unit="samples" )

		out = np.zeros(shape=(X.shape[0],self.out.shape[1]))
		#print("outshape is "+str(out.shape))
		for i in range(X.shape[0]):
			out[i] = self._predict_one_sample(X[i])
			if verbose:
				pbar.update(1)
		return out

	def predict_with_loss(self, X, Y, verbose=True):
		out = np.zeros( shape=(X.shape[0], self.out.shape[1] ) )
		loss = np.zeros( shape=X.shape[0] )
		if verbose:
			pbar = tqdm.tqdm( total=X.shape[0], unit="samples" )

		for i in range(X.shape[0]):
			out[i] = self._predict_one_sample( X[i] )
			self.loss.set_output(Y)

			loss[i] = np.sum(self.loss.feedforward(self.out))
			if verbose:
				pbar.update(1)
			# print("loss is {}".format(los))
		return [ out, loss ]

class BatchNeuralNet(NeuralNet):

	def __check_shape( actual_shape ):
		if len(actual_shape) == 1 and isinstance(self.input_shape,int):
			self.input_shape = ( self.input_shape ,)
			# print(self.input_shape)
		if len(actual_shape) != len(self.input_shape):
			if len(actual_shape) == len(self.input_shape) + 1:
				if actual_shape[1:] == self.input_shape:	
					batches = actual_shape[0]
				else:
					raise TypeError("Configured input shape ({}) is different from input shape ({})".
						format(self.input_shape,actual_shape[1:]))
			else:
				raise TypeError("Batch sizes can only be 1 dimensional")
		elif actual_shape != self.input_shape:
			raise TypeError("Configured input shape ({}) is different from input shape ({})".
						format(self.input_shape,actual_shape))

	def backpropogate( self, Y ):
		self.loss.set_output(Y)

		loss = self.loss.feedforward(self.out)
		# print("loss={}".format(loss))
		error = self.loss.backpropogate(None)

		for i in range(len(self.layers)-1,-1,-1):
			error = self.layers[i].backpropogate( error )

		for i in range(len(self.layers)):
			self.layers[i].update()
		return loss
