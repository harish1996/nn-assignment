from nn_pkgs.layer import Layer
from nn_pkgs.ConvolutionLayer.conv_layer import Convolution2D,Convolution3D,Convolution
import numpy as np

#data = np.random.normal( size=(6,6) )
#filt = np.random.uniform( size=(2,3,3) )

data = np.array( [[ 1,1,1,1,1 ],
                  [ 2,2,2,2,2 ],
                  [ 3,3,3,3,3 ],
                  [ 4,4,4,4,4 ]] )
filt = np.array( [ [[ 0,1,0],[1,2,1],[1,3,3] ],[[6,4,2],[7,4,8],[5,3,7]] ])

conv = Convolution2D( input_shape = data.shape, filter_shape = filt.shape[1:], stride=1, padding = 2 )
#print(data)
#print(filt)
out = conv.convolve( data, filt )
#print(out)

data = np.random.uniform( low= -5, high= 20, size=(3,50,50) )
filt = np.random.normal( loc=2, scale = 4, size=(6,3,5,5) )
data = data.astype(int)
filt = filt.astype(int)

conv = Convolution3D( input_shape = data.shape, filter_shape = filt.shape[1:], stride=2, padding = 2 )
print(data)
print(filt)
out = conv.convolve( data, filt )
out = out.astype(int)
print(out)

filt = np.random.normal( loc=1, scale=2, size=(3,6,7,7) )
print(filt)
filt = filt.astype(int)
conv = Convolution3D( input_shape = out.shape, filter_shape = filt.shape[1:], stride=2, padding = 1 )
out = conv.convolve( out, filt )
print(out)
