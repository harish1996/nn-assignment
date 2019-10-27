from scipy.signal import convolve2d,fftconvolve,convolve
import numpy as np 
from numpy.lib.stride_tricks import as_strided

def sliding_window_view(arr, window_shape, steps):
	
	in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]

	window_shape = np.array(window_shape)  # [Wx, (...), Wz]
	steps = np.array(steps)  # [Sx, (...), Sz]
	nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

	print(in_shape,window_shape,steps,nbytes)

	# number of per-byte steps to take to fill window
	window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
	# number of per-byte steps to take to place window
	step_strides = tuple(window_strides[-len(steps):] * steps)
	# print(window_strides,step_strides)
	# number of bytes to step to populate sliding window view
	strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

	outshape = tuple((in_shape - window_shape) // steps + 1)
	# outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
	outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
	print(window_strides,step_strides,strides,outshape)
	
	return as_strided(arr, shape=outshape, strides=strides, writeable=False)

filter = np.array([ [ [ [0,0,0],[0,-1,0],[0,0,0] ],[ [0,-1,0],[-1,7,-1],[0,-1,0] ],[ [0,0,0],[0,-1,0],[0,0,0] ] ] , [ [ [0,0,0],[0,0,0],[0,0,0] ],[ [0,0,0],[0,0,0],[0,0,0] ],[ [0,0,0],[0,0,0],[0,0,0] ] ] ])
data = np.array([[ [0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0] ],[ [0,0,0,0,0],[0,0,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,0] ],[ [0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0] ]])
data2 = np.array([ [-5,-4,-3,-4,-5],[-1,0,1,0,-1],[0,1,2,1,0],[-1,0,1,0,-1],[-5,-4,-3,-4,-5] ])
data3 = np.random.normal( size=(3,15,15) )

# print(data3.shape)
data[1,:,:] = data2
# strided = np.lib.stride_tricks.as_strided( data2, shape=(3,3), strides=(2*data2.strides[0],2*data2.strides[1]) )
strided = sliding_window_view(data[0],(3,3),(1,1))
# print(data[0])
# print(strided)
# print(filter.shape)
# print(strided[0,0,:,:,:,:])
out = np.tensordot(filter,strided,axes=([1,2,3],[3,4,5]))
# print(out.shape)
# print(out[:,0,:,:])
# out = convolve(data,filter,mode='same')
out[ out < 0 ] = 0
# print(out)