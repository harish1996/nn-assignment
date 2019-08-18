from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations

import numpy as np

data = [
		[ 1,2 ],
		[ 4,7 ],
		[ 9,2 ],
	]

data = np.array( data )

# act = activations.get("sigmoid")
# a = act()
# print(a(5))
nn = MultiLayerPerceptron( shape = (3,4,5,1), input_shape = 2, activation="sigmoid" )
out = nn.feedforward(data)

print("Input = "+str(data))
print("Output = "+str(out))
