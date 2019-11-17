import struct
import argparse
import time
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

from nn_pkgs.autoencoders import SparseAutoencoder


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
 
def form_pickle( dataset_loc, pickle_file ):
	#images = read_all_images("train-images-idx3-ubyte")
	train_x = read_all_images(dataset_loc+"train-images-idx3-ubyte")
	test_x = read_all_images(dataset_loc+"t10k-images-idx3-ubyte")

	train_x = train_x.reshape( (train_x.shape[0],-1) )
	test_x = test_x.reshape( (test_x.shape[0],-1) )
	
	total_dataset = {
		"train" : train_x,
		"test"  : test_x
	}
	
	pickle.dump( total_dataset, open(pickle_file,"wb") )
	return total_dataset

parser = argparse.ArgumentParser( description="parse" )
parser.add_argument('-e','--epochs'  , type=int  , default=3   , help="number of epochs"             	) 
parser.add_argument('-T','--train'   , type=int  , default=100 , help="number of train samples"      	)
parser.add_argument('-t','--test'    , type=int  , default=30  , help="number of test samples"       	)
parser.add_argument('-r','--rate'    , type=float, default=0.01, help="learning rate"                	)
parser.add_argument('-s','--sparsity', type=float, default=0.5 , help="Define the sparsity parameter"	)
parser.add_argument('-p','--penalty' , type=float, default=1   , help="Define penalty for sparsity"  	)
parser.add_argument('-b','--batch'   , type=int  , default=1   , help="Define batch size for training"  ) 
# parser.add_argument('-s1','--stride1', type=int  , default=1   , help="set stride for 1st cnn" )
# parser.add_argument('-s2','--stride2', type=int  , default=1   , help="set stride for 2nd cnn" )
args = parser.parse_args()

v_lr = args.rate
n_epochs = args.epochs
train_n = args.train
test_n = args.test
v_sparsity = args.sparsity
v_penalty = args.penalty
n_batch = args.batch
# st1 = args.stride1
# st2 = args.stride2
dataset_loc = "./test/conv_net/data/"

ustring = "{}R_{}E_{}TR_{}TE_{}SP_{}P_{}B_{}".format(v_lr,n_epochs,train_n,test_n,v_sparsity,v_penalty,
	n_batch,time.strftime("%d_%m_%y_%H_%M_%S"))
print(ustring)
pickle_file = "autoencoder_datasets.pickle"

try:
	print( "Reading pickle {}".format(pickle_file) )
	total_dataset = pickle.load( open(pickle_file,"rb") )
except FileNotFoundError:
	print( "Forming pickle {}".format(pickle_file))
	total_dataset = form_pickle( dataset_loc, pickle_file )

train_x = total_dataset["train"][:train_n]
# train_y = total_dataset["try"][:train_n]
test_x = total_dataset["test"][:test_n]
# test_y = total_dataset["tey"][:test_n]

scaler = StandardScaler()
scaled_train_x = scaler.fit_transform( train_x )
scaled_test_x = scaler.transform( test_x )

encoder = SparseAutoencoder( input_size=scaled_train_x.shape[1], hidden_size=40*40, lr=v_lr, 
				sparsity=v_sparsity, penalty=v_penalty )

encoder.fit( scaled_train_x, epochs=n_epochs, bs=n_batch, verbose=False )


reconstructed, loss = encoder.encode_decode( scaled_test_x )

text = "Average reconstruction error = {}".format(np.mean(loss))
print( text )

fl = open("report_"+ustring,"w")
fl.write( text )
fl.close()
