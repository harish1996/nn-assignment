import pickle
import struct
import numpy as np

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

dataset_loc = "./test/conv_net/data/"

# encoder = OneHotEncoder()

# images = read_all_images("train-images-idx3-ubyte")
# train_x = read_all_images(dataset_loc+"train-images-idx3-ubyte")
train_y = read_all_labels(dataset_loc+"train-labels-idx1-ubyte")
# test_x = read_all_images(dataset_loc+"t10k-images-idx3-ubyte")
test_y = read_all_labels(dataset_loc+"t10k-labels-idx1-ubyte")

print( np.unique(train_y, return_counts=True ))
