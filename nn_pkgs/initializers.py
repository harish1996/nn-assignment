import numpy as np 

def XavierNormal( size, loc, in_size=None, out_size=None, **kwargs ):
	
	if in_size is None or out_size is None:
		denom = np.sum(size)
	else:
		denom = np.sum(in_size) + np.sum( out_size )

	std = np.sqrt( 2 / denom )
	return Normal( size=size, loc=loc, scale= std)

def Normal( size, loc, scale, **kwargs ):
	return np.random.normal( size=size, loc=loc, scale= scale )

def Zeros( size, **kwargs ):
	return np.zeros( shape=size )

def Uniform( size, loc, scale, **kwargs )
	return np.random.uniform( size=size, low=loc-scale, high=loc+scale )

def XavierUniform( size, loc, in_size=None, out_size=None, **kwargs ):
	if in_size is None or out_size is None:
		denom = np.sum(size)
	else:
		denom = np.sum(in_size) + np.sum( out_size )

	std = np.sqrt( 6 / denom )

	return Uniform( size=size, loc=loc, scale=std )

glorot_normal = xavier_normal = XavierNormal
normal = Normal
zeros = Zeros
uniform = Uniform 
glorot_uniform = xavier_uniform = XavierUniform 

def get(identifier):
	d = globals()
	if identifier in d:
		return d[identifier]
	else:
		raise ValueError("{} is not a valid Initializer".format(identifier))
