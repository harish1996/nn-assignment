#!/usr/bin/env python
# coding: utf-8

# # PCAN ( Principal Component Analysis Network )
# This is a unsupervised network , whose weights on training, converges to the principal components of the inputs. The first node's weights converges to the first principal component and the 2nd one to the 2nd component and so on.  

# ## Exception
# The below exception is used in the code to throw the only exception that the module throws. This gives out a meaningful message with regards to the context without any use of prints

# In[1]:


class InErr(Exception):
    def __init__(self, actual_size ):
        self.desired = actual_size
        
    def __str__(self):
        return "The desired size is "+str(self.desired)+" and the received size is "+str(self.got) 
    
    def gotval(self,got):
        self.got = got


# ## PCAN
# The below class is the class for a PCAN. This is useful to build any PCAN network with any configuration. The important functions are
# 
# `PCAN(input_dim,output_dim=1)` - Creates a PCAN with input dimension = `input_dim` and output_dimension i.e. principal components = `output_dim`
# 
# `feed_one_row(inp,lr)` - Feeds to the network one row of input and also adjusts the weights accordingly.
# 
# `feed(inp,lr=0.01,epochs=100)` - Feeds the given input `inp` to the PCAN. `lr` mentions the learning rate and the model is repeatedly trained for `epochs` epochs.
# 
# `get_components()` - Returns the weight matrix of the PCAN

# In[33]:


class PCAN:
    def __init__(self, input_dim, output_dim=1, _debug=False, _debug_block=None, _debug_allow=None ):
        self.idim = input_dim
        self.odim = output_dim
        self.weights = np.random.normal( size=(self.idim,self.odim) )
        self.excep = InErr( self.idim )
        self._d = _debug
        if(self._d):
            if( _debug_block and _debug_allow ):
                raise Exception("_debug block and _debug allow cannot be used together.")
            else:
                self._db = _debug_block
                self._da = _debug_allow
    
    def __dprint(self,print_string,ident):
        if(self._d):
            if(self._db and ident in self._db):
                pass
            else:
                if( not(self._da) or (self._da and ident in self._da)):
                    print(str(ident)+":  "+print_string)
            
    def feed_one_row(self, inp, lr):
        if( inp.shape[0] != self.idim ):
            self.excep.gotval(inp.shape[0])
            raise self.excep
        
        self.__dprint("weights "+str(self.weights),0)
        y = inp.dot( self.weights )
        self.__dprint("output "+str(y),1)
        
        w_y = self.weights * y
        self.__dprint("W.Y = "+str(w_y),2)
        
        cum_sum_w_y = np.cumsum(w_y,axis=1)
        self.__dprint("w.Y_cum_sum = "+str(cum_sum_w_y),3)
        
        y_cs_wy = y * cum_sum_w_y
        self.__dprint("yj.w.Y_cum_sum = "+str(y_cs_wy),4)
        
        inp2d = inp.reshape(-1,1)
        y2d = y.reshape(1,-1)
        y_x = inp2d.dot( y2d )
        self.__dprint("y_x = "+str(y_x),5)
        
        del_w = lr * ( y_x - y_cs_wy )
        self.__dprint("del_w = "+str(del_w),6)
        
        self.weights += del_w
        
    def feed(self, inp, lr=0.01, epochs = 100):
#         print("Initial weights "+str(self.weights))
        for i in range(epochs):
            for row in range(inp.shape[0]):
                self.feed_one_row(inp[row,:],lr)
#         print("Final weights "+str(self.weights))
#         y = inp.dot( self.weights )
#         print("output "+str(y))
        
#         self.transformer = np.linalg.inv(self.weights)

    def get_components(self):
        return self.weights.T
    


# In[41]:


import numpy as np
import pandas as pd

ina = np.array([[1,2,6],[2,4,244],[3,243,16],[243,8,22]])
# inp = np.array([[1,2],[2,4],[3,6],[4,8]])
pca = PCAN( input_dim = 3, output_dim= 1, _debug=True )
pca.feed(ina, epochs = 5)
print(pca.get_components())


# ## Actual PCA
# Instead of using PCAN, the below uses PCA from sklearn to test whether the predicted PCA is the same as the correct one.

# In[21]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# a = np.array( [[1,2,3],[4,5,6]] )
# b = np.array( [[5,3,2],[1,1,1],[5,2,8]] )
# %timeit np.sum(a*b,axis=1)
# print(np.inner1d(a,b))
# print(np.einsum('ij,ij->i',a,b))
# print(np.dot(a,b.T))

# inp = np.array([[1,2],[2,4],[3,6],[4,8]])
ina = np.array([[1,2,3],[2,4,6],[3,6,9],[4,8,12]])
pca = PCA( n_components = 2 )
pca.fit(ina)
print(pca.components_)


# In[9]:



for j in range(b.shape[0]):
    print(b[j,:].shape)


# In[ ]:




