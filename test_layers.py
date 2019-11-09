from nn_pkgs.ConvolutionLayer.conv_layer import ConvolutionalLayer
import numpy as np
from nn_pkgs.loss import MSE

data = np.random.uniform( low= -5, high= 20, size=(3,50,50) )

l1 = ConvolutionalLayer( data.shape, filter_dim=(3,3,3), n_filters=5, stride=1, padding=2 )
l2 = ConvolutionalLayer( l1.output_dim, filter_dim=(l1.output_dim[0],5,5), n_filters= 10, stride = 2, padding = 0 )
loss = MSE()

rand = np.random.uniform( low=3, high=10, size=l2.output_dim )
loss.set_output( rand )

inter = l1.feedforward(data)
print("intermediate output: "+str(inter.shape))
out = l2.feedforward(inter)
print("final: "+str(out.shape))
lo = loss.feedforward( out )

grad = loss.backpropogate( None )
print("grad: "+str(grad))
inter = l2.backpropogate( grad )
print("intermediate gradient: "+str(inter))
dummy = l1.backpropogate(inter)
print("backpropogated fully")

print(l1.F)
print(l2.F)
l1.update()
l2.update()
print("after update")
print(l1.F)
print(l2.F)
# print(lo)