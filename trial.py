from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("iris.csv")
label = LabelEncoder()
integer = label.fit_transform( df['species'] )
integer = integer.reshape( -1,1) 
encoder = OneHotEncoder( sparse=False )
Y = encoder.fit_transform( integer )


# act = activations.get("sigmoid")
# a = act()
# print(a(5))

df = df.drop('species',axis=1)
data = df.values

nn = MultiLayerPerceptron( shape = (3,2,Y.shape[1]), input_shape = data.shape[1], activation="sigmoid" )
out = nn.fit(data,Y,epochs=100, bs = 5)
# out = nn.feedforward(data)
# nn.backpropogate( output )

print("Input = "+str(data))
preds = np.argmax( out, axis=1 )
print("Output= "+str(preds))
# print("Output = "+str(out))
