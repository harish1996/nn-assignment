import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

def generate( samples ):
	mu_1 = [ 5, 20 ]
	mu_2 = [ 20, 40 ]
	var = [ 1,3 ]

	# df = pd.DataFrame()

	c = [ ]
	c.append( np.random.normal( mu_1, var, (samples,2) ) )
	c.append( np.random.normal( mu_2, var, (samples,2) ) ) 
	c[0] = np.append( c[0], np.zeros(samples).reshape( (-1,1)) , axis=1 )
	c[1] = np.append( c[1], np.ones(samples).reshape( (-1,1)), axis=1 )
	c = np.append( c[0], c[1], axis=0 )
	c = np.random.permutation(c)

	df = pd.DataFrame( c, columns=["axis1", "axis2", "class" ] )
	df["class"] = pd.to_numeric( df["class"], downcast="integer" )
	# c[0].append( np.random.normal( mu_x[0], var_x, samples ) )
	# c[0].append( np.random.normal( mu_y[0], var_y, samples ) )
	# c[1].append( np.random.normal( mu_x[1], var_x, samples ) )
	# c[1].append( np.random.normal( mu_y[1], var_y, samples ) )

	return df

if __name__ == "__main__":
	samples = 1000
	a = generate( samples )
	fig , ax = plt.subplots()
	# print(a[0].shape)
	ax.scatter(a["axis1"], a["axis2"] )


	# ax.scatter(a[1][:,0], a[1][:,1] )
	# np.savetxt("2C2D.csv",a,delimiter=",")
	# print(a)
	a.to_csv("2C2D.csv",sep=",",header=None,index=None)
	plt.show()
