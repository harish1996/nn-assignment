import numpy as np
import struct
import timeit

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

def read_all_images( filename ):
	import numpy as np
	import struct
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

dataset_loc = "./test/conv_net/data/"

# a = timeit.timeit( 'read_all_images(dataset_loc+"train-images-idx3-ubyte")', number = 1, setup='import numpy as np; import struct;from __main__ import read_all_images; dataset_loc = "./test/conv_net/data/"' )
# b = timeit.timeit( 'read_n_images(dataset_loc+"train-images-idx3-ubyte",60000)', number = 1, setup='import numpy as np; import struct; from __main__ import read_n_images;  dataset_loc = "./test/conv_net/data/"' ) 
train_x = read_all_images(dataset_loc+"train-images-idx3-ubyte")
t_x = read_n_images(dataset_loc+"train-images-idx3-ubyte",60000)
# test_x = read_all_images(dataset_loc+"t10k-images-idx3-ubyte")

print( train_x )
print( t_x )
print( (train_x == t_x).all() )