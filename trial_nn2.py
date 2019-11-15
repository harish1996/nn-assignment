from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations
from nn_pkgs import datasets
from nn_pkgs.loss import MSE,BCE
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
import argparse
import time
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
   
#dataset_loc = "./test/conv_net/data/"
#
#encoder = OneHotEncoder()
#
## images = read_all_images("train-images-idx3-ubyte")
#train_x = read_all_images(dataset_loc+"train-images-idx3-ubyte")
#train_y = read_all_labels(dataset_loc+"train-labels-idx1-ubyte")
#test_x = read_all_images(dataset_loc+"t10k-images-idx3-ubyte")
#test_y = read_all_labels(dataset_loc+"t10k-labels-idx1-ubyte")
#
#train_y = encoder.fit_transform( train_y.reshape((-1,1)) )
#test_y = encoder.fit_transform( test_y.reshape((-1,1)) )
#train_y =  train_y.todense().getA()
#test_y = test_y.todense().getA()
#
#total_dataset = {
#	"trx":train_x,
#	"try":train_y,
#	"tex":test_x,
#	"tey":test_y
#}
#
#pickle.dump( total_dataset, open("datasets.pickle","wb") )
#exit()
#
parser = argparse.ArgumentParser( description="parse" )
parser.add_argument('-e','--epochs'    , type=int  , default=3   , help="number of epochs"                  )
parser.add_argument('-T','--train'     , type=int  , default=100 , help="number of train samples"           )
parser.add_argument('-t','--test'      , type=int  , default=30  , help="number of test samples"            )
parser.add_argument('-r','--rate'      , type=float, default=0.01, help="learning rate"                     )
parser.add_argument('-s1','--stride1'  , type=int  , default=1   , help="set stride for 1st cnn"            )
parser.add_argument('-s2','--stride2'  , type=int  , default=1   , help="set stride for 2nd cnn"            )
parser.add_argument('-nf1','--nfilter1', type=int  , default=8   , help="set number of filters for 1st cnn" )
parser.add_argument('-nf2','--nfilter2', type=int  , default=8   , help="set number of filters for 2nd cnn" )
args = parser.parse_args()

v_lr = args.rate
n_epochs = args.epochs
train_n = args.train
test_n = args.test
st1 = args.stride1
st2 = args.stride2
nf1 = args.nfilter1
nf2 = args.nfilter2

ustring = "{}R_{}E_{}TR_{}TE_{}{}S_{}_{}F_{}".format(v_lr,n_epochs,train_n,test_n,st1,st2,nf1,nf2,time.strftime("%d_%m_%y_%H_%M_%S"))
print(ustring)

# print(v_lr,n_epochs,train_n,test_n,st1,st2)
# exit()
total_dataset = pickle.load( open("datasets.pickle","rb") )
train_x = total_dataset["trx"][:train_n]
train_y = total_dataset["try"][:train_n]
test_x = total_dataset["tex"][:test_n]
test_y = total_dataset["tey"][:test_n]

print(test_x.shape)
## Constructing the network
network = NeuralNet( train_x.shape[1:] )
layer = ConvolutionalLayer( train_x.shape[1:], filter_dim=(5,5), n_filters=nf1, stride=st1, padding=0, lr=v_lr )
network.add(layer)
layer = activations.Sigmoid( layer.output_dim )
network.add(layer)
layer = ConvolutionalLayer( layer.output_shape, filter_dim=(nf1,3,3), n_filters=nf2, stride=st2, padding=0, lr = v_lr )
network.add(layer)
layer = activations.Sigmoid( layer.output_dim )
network.add(layer)
layer = MaxPoolingLayer( layer.output_shape, window_shape=(3,3) )
network.add(layer)
layer = UnwrapLayer( layer.output_dim )
network.add(layer)
layer = SinglePerceptronLayer( layer.output_shape[1], nodes=train_y.shape[1], lr = v_lr )
network.add(layer)
layer = activations.Softmax( layer.output_shape )
network.add(layer)
network.add_loss(BCE())

#print(type(train_y))
out = network.fit( train_x, train_y, epochs=n_epochs )

pred_y = network.predict( test_x )
pred_train_y = network.predict( train_x )

py = np.argmax( pred_y, axis = 1 )
ty = np.argmax( test_y, axis = 1 )
pty = np.argmax( pred_train_y, axis= 1 )
aty = np.argmax( train_y, axis = 1 )

fl = open("report_"+ustring,"w")
#print(py,ty)
fl.write(" Test report")
fl.write( classification_report( ty, py ) )
fl.write(" Train report" )
fl.write( classification_report( pty, aty ) )
fl.close()

pickle.dump( network, open("model_"+ustring+".pickle","wb") )
