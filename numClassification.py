# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:12:53 2020

@author: Jason Jia UniMelb ID: 1065073
"""

# =============================================================================
# In this script,i used deep learning models to learn the visual and audio data 
# to classify the movie genres. I developed a baseline model using MLP and 
# benchmarked against a fully-connected sequential model using Keras library.
# =============================================================================

import os 
# check current working directory
os.getcwd()
# set up your path
os.chdir(r'C:\\Users\\Jason\\Documents\\Study Temp\\MachineLearning\\ass2\\data\\data\\Assignment2')  

# Libraries import
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from tensorflow.keras import utils as np_utils
import tensorflow as tf  #need to install tensorflow first 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
#import pydot, pydotplus, GraphViz
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score
from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot


#Read feature files
train_feat = pd.read_csv("train_features.tsv", sep = "\t")
test_feat = pd.read_csv("NEW_test_features.tsv", sep = "\t")
valid_feat = pd.read_csv("valid_features.tsv", sep = "\t")

#remove "YTId" (youtube link)
train_feat = train_feat.drop(["YTId"], 'columns')
test_feat = test_feat.drop(["YTId"], 'columns')
valid_feat = valid_feat.drop(["YTId"], 'columns')


#Read labels
train_label = pd.read_csv("train_labels.tsv", sep = "\t")
valid_label = pd.read_csv("valid_labels.tsv", sep = "\t")

##Extract unique labels 
a = train_label[["genres"]]
a.iloc[:,0].values.tolist()
labels = np.unique(a)

##Data cleaning 
#test if movie ID is unique 
b = train_label.movieId.unique()
#b.count()  #Good, all traing ID are unique, no duplicates

#any nulls?
train_feat.isnull()
train_label.isnull()

#Check the data types of training features 
train_feat.dtypes
train_label.dtypes

#Data briefing 
train_feat.info()
c = train_feat.describe()

#unique values for each var.
train_feat.nunique() #among all of features, year has 109 unique values which 
#makes sense, tag have 3487 values, remember this 
    
#We have 3 main types of features: 
    #movieID, movieTiltle (string), year (int), Tag(text, categorical), learnt in the other script
    #audio: (avg1-avg107, continuous float)
    #visual:(ivec1-ivec20, continuous float)
        
##N.N. model

#Subset numerical data 
train_feat_num = train_feat.iloc[:, 4:131]
valid_feat_num = valid_feat.iloc[:, 4:131]

#Subset audio and visual 
train_audio = train_feat_num.iloc[:,0:107]
train_visual = train_feat_num.iloc[:, 107:131]

valid_audio = valid_feat_num.iloc[:,0:107]
valid_visual = valid_feat_num.iloc[:, 107:131]

# Labels: drop MovieId
train_label = train_label.iloc[:,-1]
valid_label = valid_label.iloc[:,-1]

# Turn to numpy ndarry
train_feat_arr = train_feat_num.values
valid_feat_arr = valid_feat_num.values


#label encoding 
# Make copy to avoid changing the orginal data
label_y_train = train_label.copy()
label_y_valid = valid_label.copy()

# Apply label encoder to the gerer column with categorical data 
label_encoder = LabelEncoder()
label_y_train= label_encoder.fit_transform(train_label)
label_y_valid = label_encoder.fit_transform(valid_label)


train_y = np_utils.to_categorical(label_y_train)
valid_y = np_utils.to_categorical(label_y_valid)

#label unique classes 
label_encode = label_encoder.fit_transform(labels)

#############################################################
# MLP model 

MLPC = MLPClassifier()
mlp = MLPClassifier(hidden_layer_sizes = (2,18), solver = "adam", learning_rate = 'adaptive')

epochs = 100

for i in range(epochs): 
    mlp.partial_fit(train_visual, label_y_train, label_encode)
    print(mlp.score(train_visual, label_y_train))
    print(mlp.predict(train_visual))

##############################################################
# Keras layer - Visual
input_dim = len(train_visual.columns)

visual_model =  Sequential()
visual_model.add(Dense(22, input_dim = input_dim , activation = 'relu'))
visual_model.add(Dense(20, activation = 'relu'))
visual_model.add(Dense(20, activation = 'relu'))
visual_model.add(Dense(18, activation = 'softmax'))
    
visual_model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

visual_model.fit(train_visual, train_y, epochs = 100 , batch_size = 100)

#score = visual_model.evaluate(valid_visual, label_y_valid, verbose=0)  #combination of the loss and the accuracy

y_pred = visual_model.predict_classes(valid_visual)

f1_score(label_y_valid, y_pred, average ="micro")

precision_score(label_y_valid, y_pred, average ="micro")

###################################################
 #kera audio
input_dim2 = len(train_audio.columns)

audio_model =  Sequential()
audio_model.add(Dense(120, input_dim = input_dim2 , activation = 'relu'))
audio_model.add(Dense(60, activation = 'relu'))
audio_model.add(Dense(60, activation = 'relu'))
audio_model.add(Dense(18, activation = 'softmax'))

audio_model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

audio_model.fit(train_audio, train_y, epochs = 100 , batch_size = 100)

score2 = audio_model.evaluate(valid_audio, valid_y, verbose=0)

y_pred2 = audio_model.predict_classes(valid_audio)

f1_score(label_y_valid, y_pred2, average ="micro")

#plot_model(audio_model, to_file = "model_plot.png", show_shapes = True, show_layer_names = True)

#more evaluation
history = audio_model.fit(train_audio, train_y, validation_data=(valid_audio, valid_y), epochs=40, verbose=0)

_, train_acc = audio_model.evaluate(train_audio, train_y, verbose=0)
_, valid_acc = audio_model.evaluate(valid_audio, valid_y, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, valid_acc))

# plot loss during training
pyplot.subplot(111)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()




