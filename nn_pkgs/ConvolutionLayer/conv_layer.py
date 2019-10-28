import numpy as np
from .. import activations
from ..layer import Layer
from numpy.lib.stride_tricks import as_strided
class Convolution():
	def __init__(self, input_shape, filter_shape, stride, padding ):

		self.input_shape = np.array( input_shape )
		self.input_shape[self.input_shape.shape[0]-2:] += 2* padding

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

		# print(self.in_shape,self.window_shape,self.window_strides,self.step_strides,self.outshape)
		self.input_shape[self.input_shape.shape[0]-2:] -= 2* padding
		self.input_shape = tuple(self.input_shape)
		
		self.padding = padding		


	def generate_windows(self, arr):

		nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

		# number of bytes to step to populate sliding window view
		strides = tuple(int(i) * nbytes for i in self.step_strides + self.window_strides)
		# print(strides,self.outshape)
		return as_strided(arr, shape=self.outshape, strides=strides, writeable=False)

	def pad(self,arr,padding):
		pass

	def gen_strides(self,stride):
		pass

	def zero_pad(self,in_shape,padding):
		padded_shape = np.array(in_shape)
		padded_shape[padded_shape.shape[0]-2:] += 2*padding
		padded_array = np.zeros( shape=padded_shape )
		return padded_array

class Convolution2D(Convolution):
	
	def convolve(self, arr, filters):
		if( arr.shape != self.input_shape ):
			raise TypeError("Input Shape must match with the initialized shape. Initialized:"+str(self.input_shape)+
				" Given:"+str(arr.shape))
		filter_shape = filters.shape[1:]
		if( ( filter_shape != self.window_shape ).all() ):
			raise TypeError("Filtered Shape must match with the initialized shape. Initialized:"+str(self.window_shape)+
				" Given:"+str(filter_shape))
		new_arr = self.pad(arr,self.padding)
		# print(new_arr.shape)
		# print(new_arr)
		windows = self.generate_windows(new_arr)
		# print(filters)
		# print(windows)
		out = np.tensordot(filters,windows,axes=([1,2],[2,3]))
		# print(out.shape)
		return out

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

	def pad(self,arr,padding):
		padded_array = self.zero_pad(arr.shape,padding)
		padded_array[padding:padding+arr.shape[0],padding:padding+arr.shape[1]] = arr
		return padded_array

class Convolution3D(Convolution):
	
	def convolve(self, arr, filters):
		if( arr.shape != self.input_shape ):
			raise TypeError("Input Shape must match with the initialized shape. Initialized:"+str(self.input_shape)+
				" Given:"+str(arr.shape))
		filter_shape = filters.shape[1:]
		if( (filter_shape != self.window_shape).all() ):
			raise TypeError("Filtered Shape must match with the initialized shape. Initialized:"+str(self.window_shape)+
				" Given:"+str(filter_shape))

		new_arr = self.pad(arr,self.padding)
		windows = self.generate_windows(new_arr)
		# print(windows)
		out = np.tensordot(filters,windows,axes=([1,2,3],[3,4,5]))
		print(out.shape)
		return out[:,0,:,:]

	def pad(self,arr,padding):
		padded_array = self.zero_pad(arr.shape,padding)
		# print(padded_array.shape)
		padded_array[:,padding:padding+arr.shape[1],padding:padding+arr.shape[2]] = arr
		return padded_array

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
	def __init__(self, input_dim,filter_dim, stride, padding = 0):

		self.input_dim = input_dim
		
		if len(filter_dim) != len(input_dim):
			raise TypeError("Number of dimensions of the input and filter must match.")
		if len(filter_dim) == 3 and filter_dim[2] != input_dim[3]:
			raise TypeError("The 3rd dimension of the filter and input must be of same length")

		self.padding = padding
		self.filter_dim = filter_dim
		self.F = np.random.normal( shape=filter_dim )
		self.stride = stride

	def feedforward(self, X):
		pass

	def backpropogate(self, error):
		pass

	def update(self):
		pass


