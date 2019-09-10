from keras.datasets import mnist
from keras import layers
from keras import models
from matplotlib import pyplot as plt
from keras.utils import to_categorical

"""Deep learning model to categorize handwritten digits using mnist dataset"""
(train_data,train_label),(test_data,test_label)=mnist.load_data()
print(train_data.shape)  
"""Check the shape of train data and test data"""
print(test_data.shape)
"""Train data consists of 60000 digit images. Test data consists of 10000 digit images."""

"""Display an example in train data
Each example is an image of size 28*28 so its feature is a list of 28*28 integers indicating the value of 28*28 pixels"""
plt.imshow(train_data[0],cmap=plt.cm.get_cmap("binary"))
print(train_label[0])
"""Before feeding data into the model, it is necessary to process.
We turn train data into 2D tensor consisting of (example,features)"""
train_data=train_data.reshape(60000,28*28)
train_data=train_data.astype("float32")/255 
"""Scale the data and change the type from unit8 to float32 as model works with float"""
test_data=test_data.reshape(10000,28*28)
test_data=test_data.astype("float32")/255 
"""There is a great disparity in accuracy results achieved by train data with scaling and train data without scaling. 
In my experiment,the latter is around 80% while the former reaches 98%
There are severals reasons for this:
- Large input data results in large learned weights, which leads to an unstable model. 
In other words, it is sensitive to input values.Small weights are preferable 
- In my first layer, the activation function is relu while the initialized weights are small, thus scaling preserves data trait. """

"""Encode the label. If I keep the original format, I will feed a list of integers to the model. Number 1 has label 1,etc
Example: The true label result is (1,5,6) , the model infers that (1,5,7) is closer to the truth than (1,5,9) based on distance
even though there is 1 wrong prediction in each case.
5 is encoded as [0 0 0 0 1 0 0 0 0 0 ]"""

train_label=to_categorical(train_label)
test_label=to_categorical(test_label)

model=models.Sequential()
"""Linear stack of layers"""
model.add(layers.Dense(512,input_shape=(28*28,),activation="relu"))
"""The first hidden unit receives the train data and has activation function relu""" 
model.add(layers.Dense(10,activation="softmax"))
"""The output unit ouputs a 10D vector by function softmax"""
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
"""The optimizer rmsprop performs better than pure sgd.
Due to the nature of softmax, loss function based on maximum likelihood is a good choice
Metrics doesn't contribute anything but represents the final quality of trained model."""
model.fit(train_data,train_label,epochs=5,batch_size=128)
"""Train model. Batch size is the size of data that loss function works on and sends feedback to the optimizer."""

model.evaluate(test_data,test_label)
"""Test the model. The generalization is quite satisfying."""

pred=model.predict(test_data[0:1])
print(pred)
print(test_label[0])
"""Given an input, softmax produces the probabilities of belonging to labels
The above is a true prediction"""
