import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math

# def generate( samples ):
# 	mu_1 = [ 5, 20 ]
# 	mu_2 = [ 20, 40 ]
# 	var = [ 1,3 ]

# 	# df = pd.DataFrame()

# 	c = [ ]
# 	c.append( np.random.normal( mu_1, var, (samples,2) ) )
# 	c.append( np.random.normal( mu_2, var, (samples,2) ) ) 
# 	c[0] = np.append( c[0], np.zeros(samples).reshape( (-1,1)) , axis=1 )
# 	c[1] = np.append( c[1], np.ones(samples).reshape( (-1,1)), axis=1 )
# 	c = np.append( c[0], c[1], axis=0 )
# 	c = np.random.permutation(c)

# 	df = pd.DataFrame( c, columns=["axis1", "axis2", "class" ] )
# 	df["class"] = pd.to_numeric( df["class"], downcast="integer" )
# 	# c[0].append( np.random.normal( mu_x[0], var_x, samples ) )
# 	# c[0].append( np.random.normal( mu_y[0], var_y, samples ) )
# 	# c[1].append( np.random.normal( mu_x[1], var_x, samples ) )
# 	# c[1].append( np.random.normal( mu_y[1], var_y, samples ) )

# 	return df
def add_classes_onehot( dataset, total_classes, this_class_id ):
		ret = [ dataset ]
		for i in range(total_classes):
			if i != this_class_id:
				ret.append( np.zeros( (dataset.shape[0],1) ) )
			else:
				ret.append( np.ones( (dataset.shape[0],1) ) )
		return np.hstack( ret )

def add_classes_binary( dataset, total_classes, this_class_id ):
	
	ret = [ dataset ]
	phy_classes = math.ceil(math.log2(total_classes-1))
	bitwise = 1
	for i in range(phy_classes):
		if( this_class_id & bitwise ):
			ret.append( np.zeros( (dataset.shape[0],1) ) )
		else:
			ret.append( np.ones( (dataset.shape[0],1) ) )
	return np.hstack( ret )

def generate_Nd( N, samples, means, vars ):

	# assert( len(samples) == 2 ),"Sample size must be 2 valued"
	classes = len(samples)

	dataset = []


	for i in range(classes):
		assert( len(means[i]) == N or len(vars[i]) == N ),"Mean and variance should be specified for all classes"
		
		# print(means[i],samples[i],vars[i])
		dat = np.random.normal( loc=means[i], size=(samples[i],N), scale=vars[i] )
		dataset.append( add_classes_onehot(dat, classes, i) )
	dataset = np.vstack(dataset)
	return dataset


# def generate3d( samples, classes, means, vars ):

# 	# samples = 1000
# 	# assert( len(samples) == 3 ), "Sample size must be 3 valued"

# 	dataset = []

# 	for i in range(classes):
# 		assert( len(means[i]) == 3 or len(vars[i]) == 3 ),"Mean and variance should be specified for all classes and variables"
# 		dat = np.random.normal( loc=means[i], size=(samples[i],3), scale=vars[i] ) 
# 		dataset.append( add_classes( dat, classes, i ) )
# 		# dataset = np.append( dataset, np.random.normal( loc=means[i], size=(samples[i],3), scale=var[i] ), axis=0 )
# 	dataset = np.vstack( dataset )
# 	return dataset
# 	# for i in range(classes):


# 	# dataset = np.random.normal( loc = (1,4,9), size=(samples,3), scale=(1,2,3) )
# 	# dataset = np.append( dataset, np.random.normal( loc=( 7,11,28), size=(samples)))


if __name__ == "__main__":
	samples = 1000

	# dataset = np.random.normal( loc = (1,4,9), size=(samples,3), scale=(1,2,3) )
	# dataset = np.append( dataset, np.random.normal( loc=( 7,11,28), size=(samples)))
	# a = generate( samples )
	a = generate_Nd( 2, (samples,samples,samples), means=[ (1,2),(50,91),(23,7) ], vars=[ (6,4),(6,7),(3,4) ])
	# fig , ax = plt.subplots()
	# print(a[0].shape)
	# ax.scatter(a[:,3], a[:,4] )


	# ax.scatter(a[1][:,0], a[1][:,1] )
	# np.savetxt("2C2D.csv",a,delimiter=",")
	# print(a)
	a = pd.DataFrame( a )
	a.to_csv("3C2D.csv",sep=",",header=None,index=None)
	# plt.show()
