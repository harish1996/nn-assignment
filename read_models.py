import struct
import argparse
import time
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

from nn_pkgs.autoencoders import SparseAutoencoder

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

def read_all_images( filename ):
	return read_n_images( filename )

def read_n_images( filename, n="all" ):
	import numpy as np
	import struct
	print("Extracting {}".format(filename))
	f = open(filename,"rb")
	header = f.read(16)
	header = struct.unpack('>4i',header)
	(magic,nimages,rsize,csize) = header
	assert(magic==2051),"invalid header."
	if isinstance(n,str):
		if n=="all":
			n = nimages
		else:
			raise ValueError("Unknown command")
	elif isinstance(n,int): 
		if n > nimages:
			raise ValueError("n ({}) is higher than the total number of images ({})".format(n,nimages))
		if n <= 0:
			raise ValueError("n ({}) should be positive integer".format(n))
	img_size = rsize*csize
	total_size = n*img_size
	buf = f.read(total_size)
	images = np.frombuffer( buf, dtype=np.uint8).astype(np.float32)
	images = images.reshape((n,rsize,csize))
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
 
parser = argparse.ArgumentParser( description="parse" )
parser.add_argument('-m','--model', type=str, required=True,help="model name")
parser.add_argument('-o','--outfile',type=str, required=True,help="outfile name")

args = parser.parse_args()
name = args.model
outfile = args.outfile

dataset_loc = "./test/conv_net/data/"

pickle_file = "autoencoder_datasets.pickle"

try:
	print( "Reading pickle {}".format(pickle_file) )
	total_dataset = pickle.load( open(pickle_file,"rb") )
except FileNotFoundError:
	print( "Forming pickle {}".format(pickle_file))
	total_dataset = form_pickle( dataset_loc, pickle_file )

train_x = total_dataset["train"]
test_x = total_dataset["test"]

scaler = MinMaxScaler()
scaled_train_x = scaler.fit_transform( train_x )
scaled_test_x = scaler.transform( test_x )

print("Loading model")
encoder = pickle.load( open(name,"rb") )

reconstructed, loss = encoder.encode_decode( scaled_test_x )

rec_rescaled = scaler.inverse_transform( reconstructed )
loss = np.abs( rec_rescaled - test_x )
text = "Average test set reconstruction error = {}".format(np.mean(loss))

reconstructed, loss = encoder.encode_decode( scaled_train_x )

rec_rescaled = scaler.inverse_transform( reconstructed )
loss = np.abs( rec_rescaled - train_x )
text += " \n Average train set reconstruction error = {}".format(np.mean(loss))
print( text )

layer = encoder.sparse_layer()
rep = layer.W[1,1:].reshape(28,28)
rep1 = layer.W[2,1:].reshape(28,28)

plt.imsave( arr=rep, fname="internal_rep"+outfile )
#normie = Normalize( rep )
print("************** Saving images **********")
plt.imsave( arr=scaled_train_x[115].reshape(28,28), fname="input_"+outfile )
plt.imsave( arr=reconstructed[115].reshape(28,28),  fname="output_"+outfile )

