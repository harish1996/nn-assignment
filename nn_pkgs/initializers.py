import numpy as np 

def XavierNormal( size, loc, in_size=None, out_size=None ):
	
	if in_size is None or out_size is None:
		denom = np.sum(size)
	else:
		denom = np.sum(in_size) + np.sum( out_size )

	std = np.sqrt( 2 / denom )
	return np.random.normal( size=size, loc=loc, scale= std)



