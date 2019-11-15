from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations
from nn_pkgs import datasets
from nn_pkgs.loss import MSE
from nn_pkgs.ConvolutionLayer.conv_layer import ConvolutionalLayer,MaxPoolingLayer,UnwrapLayer
from nn_pkgs.ConvolutionLayer.conv_layer import NeuralNet
from nn_pkgs.SinglePerceptron.slp import SinglePerceptronLayer 

import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# import pandas as pd

from sklearn.model_selection import train_test_split

import struct
# import numpy as np

def read_all_images( filename ):
        print("Extracting {}".format(filename))
        f = open(filename,"rb")
        header = f.read(16)
        header = struct.unpack('>4i',header)
        (magic,nimages,rsize,csize) = header
#         print(magic,nimages,rsize,csize)
#         print(header)
#         print("magic number = {}, num_items = {}, rows = {}, cols = {}".format(*header))
        assert(magic==2051),"invalid header."
        images = np.empty( shape=(nimages,rsize,csize) )
        img_size = rsize*csize
        total_size = nimages*img_size
        total = struct.unpack('{}B'.format(total_size), f.read(total_size))
        for inum in range(nimages):
            im = total[img_size*inum:img_size*(inum+1)]
            for row in range(rsize):
#                 r = struct.unpack('{}B'.format(header[3]), f.read(header[3]) )
                images[inum,row,:] = im[csize*row:csize*(row+1)]
#         print(images[3,:,:])
        return images
        
def read_all_labels( filename ):
    print("Extracting {}".format(filename))
    f = open(filename,"rb")
    header = f.read(8)
    header = struct.unpack('>2i',header)
    (magic,nlabels) = header
#     print(nlabels)
    assert(magic==2049),"invalid header."
    return np.array( struct.unpack('{}B'.format(nlabels), f.read(nlabels)) )
    
# dataset_loc = "./test/conv_net/data/"

# encoder = OneHotEncoder()

# # images = read_all_images("train-images-idx3-ubyte")
# train_x = read_all_images(dataset_loc+"train-images-idx3-ubyte")
# train_y = read_all_labels(dataset_loc+"train-labels-idx1-ubyte")
# test_x = read_all_images(dataset_loc+"t10k-images-idx3-ubyte")
# test_y = read_all_labels(dataset_loc+"t10k-labels-idx1-ubyte")

# train_y = encoder.fit_transform( train_y.reshape((-1,1)) )
# test_y = encoder.fit_transform( test_y.reshape((-1,1)) )
# train_y =  train_y.todense().getA()
# test_y = test_y.todense().getA()

# total_dataset = {
# 	"trx":train_x,
# 	"try":train_y,
# 	"tex":test_x,
# 	"tey":test_y
# }

# pickle.dump( total_dataset, open("datasets.pickle","wb") )
train_n = 1000
test_n = 300

total_dataset = pickle.load( open("datasets.pickle","rb") )
train_x = total_dataset["trx"][:train_n]
train_y = total_dataset["try"][:train_n]
test_x = total_dataset["tex"][:test_n]
test_y = total_dataset["tey"][:test_n]

print(test_x.shape)
## Constructing the network
network = NeuralNet( train_x.shape[1:] )
layer = ConvolutionalLayer( train_x.shape[1:], filter_dim=(5,5), n_filters=8, stride=1, padding=0 )
network.add(layer)
layer = activations.ReLU( layer.output_dim )
network.add(layer)
layer = ConvolutionalLayer( layer.output_shape, filter_dim=(8,5,5), n_filters=8, stride=1, padding=0 )
network.add(layer)
layer = activations.ReLU( layer.output_dim )
network.add(layer)
layer = MaxPoolingLayer( layer.output_shape, window_shape=(3,3) )
network.add(layer)
layer = UnwrapLayer( layer.output_dim )
network.add(layer)
layer = SinglePerceptronLayer( layer.output_shape[1], nodes=train_y.shape[1] )
network.add(layer)

network.add_loss(MSE())

print(type(train_y))
out = network.fit( train_x, train_y, epochs=3)

pred_y = network.predict( test_x )

print(pred_y,test_y)



