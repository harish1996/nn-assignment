from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd

from sklearn.model_selection import train_test_split

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

train_x, test_x, train_y, test_y = train_test_split( data, Y, test_size = 0.2 )

nn = MultiLayerPerceptron( shape = (3,4,2,Y.shape[1]), input_shape = data.shape[1], activation="sigmoid" )
out = nn.fit(train_x,train_y,epochs=500, bs = 5)
# out = nn.feedforward(data)
# nn.backpropogate( output )

pred_y = nn.predict( test_x )
# print("Input = "+str(data))
preds = np.argmax( pred_y, axis=1 )
actual_y = np.argmax( test_y, axis=1 )

print(classification_report(actual_y,preds))
print(confusion_matrix(actual_y,preds))
# print("Output= "+str(preds))
# print("Output = "+str(out))
