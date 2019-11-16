import numpy as np
from .. import activations
from ..layer import Layer
from numpy.lib.stride_tricks import as_strided
import math
import tqdm
from ..loss import Loss

class Convolution():
	def __init__(self, input_shape, filter_shape, stride, padding ):

		self.input_shape = np.array( input_shape )
		self.input_shape[self.input_shape.shape[0]-2:] += 2* padding
		self.padded_input_shape = tuple(self.input_shape)
		# print(self.padded_input_shape)
		
		self.window_shape = filter_shape
		self.stride = self.gen_strides(stride)

		# print(self.input_shape,self.window_shape,self.stride)
		
		self.in_shape = np.array(self.input_shape[-len(self.stride):])  # [x, (...), z]
		self.window_shape = np.array(self.window_shape)  # [Wx, (...), Wz]
		self.stride = np.array(self.stride)  # [Sx, (...), Sz]

		# number of per-byte steps to take to fill window
		self.window_strides = tuple(np.cumprod(self.input_shape[:0:-1])[::-1]) + (1,)
		# number of per-byte steps to take to place window
		self.step_strides = tuple(self.window_strides[-len(self.stride):] * self.stride )

		self.outshape = tuple((self.in_shape - self.window_shape) // self.stride + 1)
		# outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
		self.outshape = self.outshape + tuple(self.input_shape)[:-len(self.stride)] + tuple(self.window_shape)
		# print("Outshape"+str(self.outshape[:len(self.stride)]))

		# print(self.window_strides,self.step_strides,self.outshape)

		self.output_shape = self.outshape[:len(self.stride)]
		# print(self.in_shape,self.window_shape,self.window_strides,self.step_strides,self.outshape)
		self.input_shape[self.input_shape.shape[0]-2:] -= 2* padding
		self.input_shape = tuple(self.input_shape)
		# print(self.padded_input_shape)
		
		self.padding = padding


	def generate_windows(self, arr, writeable=False):

		nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

		# number of bytes to step to populate sliding window view
		strides = tuple(int(i) * nbytes for i in self.step_strides + self.window_strides)
		# print(strides,self.outshape)
		return as_strided(arr, shape=self.outshape, strides=strides, writeable=writeable)

	def pad(self,arr):
		pass

	def gen_strides(self,stride):
		pass

	def zero_pad(self,in_shape):
		padded_shape = np.array(in_shape)
		padded_shape[padded_shape.shape[0]-2:] += 2*self.padding
		padded_array = np.zeros( shape=padded_shape )
		return padded_array

class Convolution2D(Convolution):
	
	def convolve(self, arr, filters, biases):
		if( arr.shape != self.input_shape ):
			raise TypeError("Input Shape must match with the initialized shape. Initialized:"+str(self.input_shape)+
				" Given:"+str(arr.shape))
		filter_shape = filters.shape[1:]
		if( ( filter_shape != self.window_shape ).all() ):
			raise TypeError("Filtered Shape must match with the initialized shape. Initialized:"+str(self.window_shape)+
				" Given:"+str(filter_shape))
		if( filters.shape[0] != biases.shape[0] ):
			raise TypeError("Number of biases should be equal to number of filters")
		if( len(biases.shape) != 1 ):
			raise TypeError("Biases should be single dimensional array of floats")

		new_arr = self.pad(arr)
		# print(new_arr.shape)
		# print(new_arr)
		self.windows = self.generate_windows(new_arr)
		# print(filters)
		# print(windows.shape)
		out = np.tensordot(filters,self.windows,axes=([1,2],[2,3]))
		# print(out.shape)
		for i in range(biases.shape[0]):
			out[i] += biases[i]
		return out

	def gradient(self,error):
		# print("error shape ="+str(error.shape)+" output shape="+str(self.output_shape))
		if error.shape[1:] != self.output_shape:
			raise TypeError("The dimension of error {} is not as expected {}".format(error.shape[1:],self.output_shape[1:]))
		n_filters = error.shape[0]
		channel_size = error.shape[1:]
		# print(error.shape,self.windows.shape)
		out = np.tensordot(error,self.windows,axes=([1,2],[0,1]))
		for i in range(n_filters):
			grad_b = np.sum( error[i] )
		
		return ( out, grad_b )

	def gen_strides(self,stride):
		if isinstance(stride,(list,tuple)):
			if len(stride) == 2:
				return stride 
			else:
				raise TypeError("Stride needs to be 3 dimensional for Convolution3D")
		elif isinstance(stride,int):
			return (stride,stride)
		else:
			raise ValueError("Stride can be a list/tuple or an integer")

	def pad(self,arr):
		padded_array = self.zero_pad(arr.shape)
		padded_array[self.padding:self.padding+arr.shape[0],self.padding:self.padding+arr.shape[1]] = arr
		return padded_array

	def remove_pad(self,arr):
		return arr[self.padding:arr.shape[0]-self.padding,self.padding:arr.shape[1]-self.padding]

class Convolution3D(Convolution):
	def __init__(self,input_shape,filter_shape,stride,padding):
		if input_shape[0] != filter_shape[0]:
			raise TypeError("Filter's third dimension ( here "+str(input_shape[0])+" ) should be equal to input's third dimension ( here "
				+ str(filter_shape[0])+" )")
		super().__init__(input_shape,filter_shape,stride,padding)

	def convolve(self, arr, filters, biases):
		if( arr.shape != self.input_shape ):
			raise TypeError("Input Shape must match with the initialized shape. Initialized:"+str(self.input_shape)+
				" Given:"+str(arr.shape))
		filter_shape = filters.shape[1:]
		if( (filter_shape != self.window_shape).all() ):
			raise TypeError("Filtered Shape must match with the initialized shape. Initialized:"+str(self.window_shape)+
				" Given:"+str(filter_shape))
		if( filters.shape[0] != biases.shape[0] ):
			raise TypeError("Number of biases should be equal to number of filters"+str(filters.shape)+" and "+str(biases.shape))
		if( len(biases.shape) != 1 ):
			raise TypeError("Biases should be single dimensional array of floats")

		new_arr = self.pad(arr)
		self.windows = self.generate_windows(new_arr)
		# print(self.windows.shape)
		out = np.tensordot(filters,self.windows,axes=([1,2,3],[3,4,5]))
		# print(out.shape)
		for i in range(biases.shape[0]):
			out[i] += biases[i]

		return out[:,0,:,:]

	def gradient(self,error):
		# print("error shape ="+str(error.shape)+" output shape="+str(self.output_shape))
		if error.shape[1:] != self.output_shape[1:]:
			raise TypeError("The dimension of error is not as expected")
		n_filters = error.shape[0]
		channel_size = error.shape[1:]
		# print(error.shape,self.windows.shape)
		out = np.tensordot(error,self.windows,axes=([1,2],[1,2]))
		for i in range(n_filters):
			grad_b = np.sum( error[i] )

		return ( out[:,0,:,:], grad_b )

	def pad(self,arr):
		padding = self.padding
		padded_array = self.zero_pad(arr.shape)
		# print(padded_array.shape)
		padded_array[:,padding:padding+arr.shape[1],padding:padding+arr.shape[2]] = arr
		return padded_array

	def remove_pad(self,arr):
		padding = self.padding
		return arr[:,padding:arr.shape[1]-padding,padding:arr.shape[2]-padding]


	def gen_strides(self,stride):
		if isinstance(stride,(list,tuple)):
			if len(stride) == 3:
				return stride 
			else:
				raise TypeError("Stride needs to be 3 dimensional for Convolution3D")
		elif isinstance(stride,int):
			return (1,stride,stride)
		else:
			raise ValueError("Stride can be a list/tuple or an integer")


class ConvolutionalLayer(Layer):
	def __init__(self, input_dim,filter_dim, n_filters = 1 , stride = 1, padding = 0, lr=0.01):

		self.input_dim = input_dim
		
		if len(filter_dim) != len(input_dim):
			raise TypeError("Number of dimensions of the input and filter must match.")
		if len(filter_dim) == 3 and filter_dim[0] != input_dim[0]:
			raise TypeError("The 3rd dimension of the filter and input must be of same length")
		if n_filters <= 0:
			raise ValueError("Number of filters must be positive")
		if not isinstance(n_filters,int):
			raise TypeError("Number of filters must be a positive integer")
		if stride <= 0:
			raise ValueError("Stride must be a positive integer")
		if not isinstance(stride,int):
			raise TypeError("Stride must be a positive integer")
		if not isinstance(lr,float) or lr <= 0:
			raise TypeError("Learning rate must be a positive real value")

		self.padding = padding
		self.filter_dim = filter_dim
		self.n_filters = n_filters
		self.F = np.random.normal( size=(self.n_filters, )+ filter_dim )
		self.bias = np.random.normal( size=self.n_filters )
		self.stride = stride
		self.lr = lr

		if len(self.filter_dim) == 2:
			self.conv = Convolution2D(self.input_dim,self.filter_dim,self.stride,self.padding)
			self.output_dim = (self.n_filters,) + self.conv.output_shape
			# print(self.conv.output_shape)
		elif len(self.filter_dim) == 3:
			self.conv = Convolution3D(self.input_dim,self.filter_dim,self.stride,self.padding)
			self.output_dim = (self.n_filters,) + self.conv.output_shape[1:]
		else:
			raise ValueError("Filter should be more than 2/3 dimensional")
		 
		# print(self.output_dim)

	def feedforward(self, X):
		if X.shape != self.input_dim:
			raise TypeError("Input dimension is not matching")

		out = self.conv.convolve(X,self.F,self.bias)
		return out

	def backpropogate(self, error):
		if error.shape != self.output_dim:
			raise TypeError("Error dimension is not equal to the output dimension {} and {}".format(error.shape,self.output_dim))
		# print("conv layer error = {}".format(error))
		self.grad_F, self.grad_b = self.conv.gradient(error)

		error_gradients = np.zeros( shape=self.conv.padded_input_shape )
		windows = self.conv.generate_windows( error_gradients, writeable=True )
		if len(self.filter_dim) == 3:
			windows = windows[0]
		# print(windows.shape)
		# print(error_gradients.shape)
		# print(self.input_dim)
		# print(self.conv.windows.shape)
		for i in range(self.n_filters):
			filt = self.F[i]
			err = error[i]
			for j in range(err.shape[0]):
				for k in range(err.shape[1]):
					# print(error_gradients[0,0,-1])
					windows[j,k] += filt * err[j,k]	

		return self.conv.remove_pad(error_gradients)

	def update(self):
		self.F -= self.lr * self.grad_F
		self.bias -= self.lr * self.grad_b


class PoolingLayer(Layer):
	def __init__(self, input_shape, window_shape ):
		if len(window_shape) != 2:
			raise TypeError("Window size should be 2 dimensional")

		self.input_shape = input_shape
		self.window_shape = window_shape
		
		single_in_shape = self.input_shape[-2:]
		
		if len(self.input_shape) == 3:
			self.n_dim = self.input_shape[0]
		else:
			self.n_dim = 1

		self.output_shape = self.input_shape[:-2] + ( math.ceil(single_in_shape[0]/window_shape[0]), 
					math.ceil(single_in_shape[1]/window_shape[1]) )

		self.output_dim = self.output_shape

	def get_window(self, arr, x, y):
		x_start = x*self.window_shape[0]
		y_start = y*self.window_shape[1]
		return arr[x_start:x_start+self.window_shape[0], y_start:y_start+self.window_shape[1]]

	def feedforward( self, X ):
		pass

	def backpropogate(self, error):
		pass

	def update(self):
		pass

class MaxPoolingLayer(PoolingLayer):

	def feedforward(self, X):

		out = np.zeros( shape=self.output_shape )
		self.mem = np.zeros( shape=self.output_shape, dtype=int )

		for i in range(self.n_dim):
			for j in range(self.output_shape[-2]):
				for k in range(self.output_shape[-1]):
					window = self.get_window(X[i],j,k)
					self.mem[i,j,k] = np.argmax( window )
					out[i,j,k] = window.flat[ self.mem[i,j,k] ]

		return out

	def backpropogate(self, error):
		if error.shape != self.output_shape:
			raise TypeError("Error dimension should be equal to the output dimension")

		out = np.zeros( shape=self.input_shape )
		# print(out.shape)
		# print("max pooling error = {}".format(error))
		for i in range(self.n_dim):
			for j in range(self.output_shape[-2]):
				x_start = j * self.window_shape[0]
				for k in range(self.output_shape[-1]):
					y_start = k * self.window_shape[1]
					out_x = int(x_start + (self.mem[i,j,k]/self.window_shape[0])) - 1
					out_y = int(y_start + (self.mem[i,j,k]%self.window_shape[0])) - 1
					# print(i,out_x,out_y)
					if out_x < out.shape[1] and out_y < out.shape[2]:
						out[i,out_x,out_y] = error[i,j,k]

		return out

class AveragePoolingLayer(PoolingLayer):

	def feedforward(self,X):
		out = np.zeros( shape=self.output_shape )

		for i in range(self.n_dim):
			for j in range(self.output_shape[-2]):
				for k in range(self.output_shape[-1]):
					out[i,j,k] = np.average( self.get_window(X[i],j,k))

		return out 

	def fill_window(self, arr, x, y, fill_value):
		x_start = x*self.window_shape[0]
		y_start = y*self.window_shape[1]
		arr[x_start:x_start+self.window_shape[0], y_start:y_start+self.window_shape[1]] = fill_value
		return arr

	def backpropogate(self, error):
		if error.shape != self.output_shape:
			raise TypeError("Error dimension should be equal to the output dimension")

		out = np.zeros( shape= input_shape )
		size = self.output_shape[-2] * self.output_shape[-1]

		averaged_error = error / size

		for i in range(self.n_dim):
			for j in range(self.output_shape[-2]):
				for k in range(self.output_shape[-1]):
					# Bug... Edge case will produce problems. Unnecessary averaging in some cases
					out[i] = self.fill_window( out[i], j, k, averaged_error[i,j,k] )

		return out

class UnwrapLayer(Layer):
	def __init__(self, input_shape ):
		self.input_shape = input_shape
		self.output_shape = 1
		for i in range(len(input_shape)):
			self.output_shape *= input_shape[i]
		self.output_shape = (1,self.output_shape)

	def feedforward(self,X):
		if X.shape != self.input_shape:
			raise TypeError("Input shape is not matching with the configured size")
		return X.reshape((1,-1))

	def backpropogate(self,error):
		if error.shape != self.output_shape:
			raise TypeError("Error shape {} is not matching with the output shape {}".format(error.shape,self.output_shape))
		# print("unwrap layer error= {}".format(error))
		return error.reshape(self.input_shape)

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

	def feedforward(self,X):
		# print("length is {}".format(type(self.input_shape)))
		if len(X.shape) == 1 and isinstance(self.input_shape,int):
			self.input_shape = ( self.input_shape ,)
			# print(self.input_shape)
		if X.shape != self.input_shape:
			raise TypeError("Input shape {} is not matching with configured shape {}".format(X.shape,self.input_shape))

		feed = X

		for i in range(len(self.layers)):
			feed = self.layers[i].feedforward( feed )
	
		self.out = feed 
		return self.out

	def feedforward_upto(self, X, upto= None ):
		if X.shape != self.input_shape:
			raise TypeError(("Input shape {} is not matching with"+ 
				"configured shape {}").format(X.shape,self.input_shape))
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
			out = self._fit_one_epoch( X,Y, shuffle )
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




