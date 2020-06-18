# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:12:53 2020

@author: Jason Jia UniMelb ID: 1065073
"""
# =============================================================================
# This script primarily focuses on the textual metadata with each movie's genre
# We use NLTK package to vectorize text contents. 
# Keywords: NLP | Text Mining | TF-IDF | Logistic Regression | Grid Search | 
# Optimization
# =============================================================================

import os 
#check current working directory
os.getcwd()
#set up your path
os.chdir(r'C:\\Users\\Jason\\Documents\\Study Temp\\MachineLearning\\ass2\\data\\data\\Assignment2')  

# Import libararies
import pandas as pd 
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import linear_model, metrics
import nltk
import re
from sklearn.metrics import f1_score
from numpy import array
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import asarray
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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


# Subset the MetaData text features 
train_text = train_feat.iloc[:,0:4]
valid_text = valid_feat.iloc[:,0:4]
test_text = test_feat.iloc[:,0:4]  

#"year" default set to string, year hardly has any relavancy to the genre of a movie
# will drop it
train_text = train_text.drop(["year"], "columns")
valid_text = valid_text.drop(["year"], "columns")
test_text = test_text.drop(["year"], "columns")

# Append the genres to corresponding movieId, to create a dataset with text 
# features and genres
train_text_full = pd.merge(train_text, train_label, on = "movieId")
valid_text_full = pd.merge(valid_text, valid_label, on = "movieId")
test_text_full = test_text

# Create a dictionary of genres and their occurrence-count
all_genres = nltk.FreqDist(train_text_full["genres"])
all_genres_df = pd.DataFrame({"Genre": list(all_genres.keys()), "Count": 
                              list(all_genres.values())})

# Time to plot the occurences of each genre and do some basic EDA
g = all_genres_df.nlargest(columns ="Count", n = 18) # descending order
plt.figure(figsize = (12,15))
ax = sns.barplot(data = g, x = "Count", y = "Genre")
plt.show()


# Define Text Cleaning function : remove "," "()" "_" and whitespace
def clean(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 

    return text

# Apply clean function
train_text_full["clean_tag"] = train_text_full["tag"].apply(lambda x: clean(x))
valid_text_full["clean_tag"] = valid_text_full["tag"].apply(lambda x: clean(x))
test_text_full["clean_tag"] = test_text_full["tag"].apply(lambda x: clean(x))


# Some records in title are pure numbers, convert them to strings 
train_text_full['title'] = train_text_full['title'].fillna("").apply(str)
valid_text_full['title'] = valid_text_full['title'].fillna("").apply(str)
test_text_full['title'] = test_text_full['title'].fillna("").apply(str)

train_text_full["clean_tilte"] = train_text_full["title"].apply(lambda x: clean(x))
valid_text_full["clean_tilte"] = valid_text_full["title"].apply(lambda x: clean(x))
test_text_full["clean_tilte"] = test_text_full["title"].apply(lambda x: clean(x))


# What are the most frequent words?

# Define a word freqency extraction funtion
def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 
                           'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()

# Apply on the tag(viewer's annotation)
freq_words(train_text_full["clean_tag"], 100)

# Class imbalance
freq_words(train_label["genres"], 100)  #very imbalanced 

# We can see large amount stopwords(SW) exist that have minimal meaning. We will 
# remove them to reduce the noise to the data. 

# Import conventional stopwords
SW = set(stopwords.words("English"))

# Define stopwords removal function 
# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in SW]
    return ' '.join(no_stopword_text)

train_text_full["clean_tag"] = train_text_full["clean_tag"].apply(lambda x: remove_stopwords(x))
valid_text_full["clean_tag"] = valid_text_full["clean_tag"].apply(lambda x: remove_stopwords(x))
test_text_full["clean_tag"] = test_text_full["clean_tag"].apply(lambda x: remove_stopwords(x))

freq_words(train_text_full["clean_tag"], 100)  # many "r" and "clv", not sure its meaning
freq_words(valid_text_full["clean_tag"], 100)
freq_words(test_text_full["clean_tag"], 100)

# Covert text to features 
y_train2 = array(train_text_full["genres"])
onehot_encoder = OneHotEncoder(sparse=False)
y_train2 = y_train2.reshape(len(y_train2), 1)
y_train2 = onehot_encoder.fit_transform(y_train2)

# Label encode 
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_text_full["genres"])

    #y_valid2 = array(valid_text_full["genres"])
    #onehot_encoder = OneHotEncoder(sparse=False)
    #y_valid2 = y_valid2.reshape(len(y_valid2), 1)
    #y_valid2 = onehot_encoder.fit_transform(y_valid2)
    
y_valid = label_encoder.fit_transform(valid_text_full["genres"])

# Use TF-IDF to extract features 
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(train_text_full["clean_tag"])
xvalid_tfidf = tfidf_vectorizer.transform(valid_text_full["clean_tag"]) 
xtest_tfidf = tfidf_vectorizer.transform(test_text_full["clean_tag"])

#############################################
# Build Movie Genre Classification Model

# Logistic Regression Model 
lr = LogisticRegression()

# fit model on train data
lr.fit(xtrain_tfidf, y_train)

# make predictions for validation set
y_pred = lr.predict(xvalid_tfidf)

#use the inverse_transform( ) functionto convert the predicted arrays into movie genre tags: 
y_pred = label_encoder.inverse_transform(y_pred)

# evaluate performance f-1 score
a = f1_score(valid_text_full['genres'], y_pred, average="micro")
print(a)

## Fine tuning the model with threshold value 
# Predict probability
y_pred_prob = lr.predict_proba(xvalid_tfidf)
# Set threshold val. = 0.3
t = 0.3
y_pred_new = (y_pred_prob >= t).astype(int)
y_pred_new = onehot_encoder.inverse_transform(y_pred_new)
# Evaluation F1
f1_score(valid_text_full['genres'], y_pred_new, average="micro")  

# Grid Search 
pipe = Pipeline([("classifier", LogisticRegression())])

param_grid = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [1.0, 0.8],
    'classifier__class_weight': [None, 'balanced'],
    'classifier__n_jobs': [-1]
}

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 10, verbose = True, n_jobs=-1)

best_clf = clf.fit(xtrain_tfidf, y_train)

# Retrieve the best parameters
best_clf.best_estimator_.get_params()['classifier']

# Logiitic Model tuning with new parameters
lr2 = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

# fit model on train data
lr2.fit(xtrain_tfidf, y_train)
# make predictions for validation set
y_pred2 = lr2.predict(xvalid_tfidf) 
y_pred2 = label_encoder.inverse_transform(y_pred2)

# evaluate performance
d = f1_score(valid_text_full['genres'], y_pred2, average="micro")
print(d)   #Great, we can see improvment on the f1-Score

# Accuracy 
best_clf.score(xvalid_tfidf, y_valid)

classes = best_clf.predict(xvalid_tfidf)
print(metrics.classification_report(classes,y_valid))

test_class = best_clf.predict(xtest_tfidf)
test_class_text= label_encoder.inverse_transform(test_class)


# Output the test result
asarray(test_class_text)
np.savetxt("test_result_text.csv",test_class_text, fmt='%5s', delimiter = ",")



















