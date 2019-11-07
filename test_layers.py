from nn_pkgs.ConvolutionLayer.conv_layer import ConvolutionalLayer
import numpy as np

data = np.random.uniform( low= -5, high= 20, size=(3,50,50) )

l1 = ConvolutionalLayer( data.shape, filter_dim=(3,3,3), n_filters=5, stride=1, padding=2 )
l2 = ConvolutionalLayer( l1.output_dim, filter_dim=(l1.output_dim[0],5,5), n_filters= 10, stride = 2, padding = 0 )

inter = l1.feedforward(data)
print("intermediate output: "+str(inter.shape))
out = l2.feedforward(inter)
print("final: "+str(out.shape))