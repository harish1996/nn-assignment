from nn_pkgs.MLP.mlp import MultiLayerPerceptron
from nn_pkgs import activations
from nn_pkgs import datasets

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pandas as pd

from sklearn.model_selection import train_test_split

# df = pd.read_csv("iris.csv")
# label = LabelEncoder()
# integer = label.fit_transform( df['species'] )
# integer = integer.reshape( -1,1) 
# encoder = OneHotEncoder( sparse=False )
# Y = encoder.fit_transform( integer )


# act = activations.get("sigmoid")
# a = act()
# print(a(5))

# df = df.drop('species',axis=1)
# data = df.values

# train_x, test_x, train_y, test_y = train_test_split( data, Y, test_size = 0.2 )

print("loading dataset")
train_x, test_x, train_y, test_y = datasets.mnist( 0.8 )

print("building perceptron")
nn = MultiLayerPerceptron( shape = (20,15,train_y.shape[1]), input_shape = train_x.shape[1], activation="sigmoid" )

print("fitting model.")
out = nn.fit(train_x,train_y,epochs=40, bs = 5)
# out = nn.feedforward(data)
# nn.backpropogate( output )

print("Predicting using the model..")
pred_y = nn.predict( test_x )

# print("Input = "+str(data))
print("Deriving results from predictions..")

train_preds = np.argmax( out, axis=1 )
actual_train_y = np.argmax( train_y, axis=1 )
preds = np.argmax( pred_y, axis=1 )
actual_y = np.argmax( test_y, axis=1 )

# print(classification_report(test_y,pred_y))
# print(test_y,pred_y)
print("Training metrics")
print(classification_report(actual_train_y,train_preds))
print(confusion_matrix(actual_train_y,train_preds))

print("Testing metrics")
print(classification_report(actual_y,preds))
print(accuracy_score(actual_y,preds))
print(confusion_matrix(actual_y,preds))
# print("Output= "+str(preds))
# print("Output = "+str(out))
