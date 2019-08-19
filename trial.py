from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations

import numpy as np

data = [
		[ 1,2 ],
		[ 4,7 ],
		[ 9,2 ],
	]
output = [  [1],
	    [6],
	    [5]
	]
data = np.array( data )
output = np.array( output )
# act = activations.get("sigmoid")
# a = act()
# print(a(5))
nn = MultiLayerPerceptron( shape = (3,1), input_shape = 2, activation="sigmoid" )
out = nn.fit(data,output,epochs=4)
# out = nn.feedforward(data)
# nn.backpropogate( output )

print("Input = "+str(data))
print("Output = "+str(out))
