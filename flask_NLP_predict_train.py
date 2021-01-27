# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:01:17 2020

@author: Hemakshi Pandey
"""


# NLP with BAG OF MODEL using SUPPORT VECTOR MACHINE


## Importing the libraries
import numpy as np  
#NumPy is a python library used for working with arrays.

import pandas as pd   
#They are used in Python to deal with data analysis and manipulation. To put it in simpler words, Pandas help us to organize data and manipulate the data by putting it in a tabular form.

import nltk
# NLTK is a leading platform for building Python programs to work with human language data.

import pickle 
#Comes handy to save complicated data.Python pickle module is used for serializing and de-serializing python object structures.

import re
#This module provides regular expression matching operations

from nltk.corpus import stopwords
nltk.download('stopwords')
# One of the major forms of pre-processing is to filter out useless data. 
#In natural language processing, useless words (data), are referred to as stop words.


nltk.download('wordnet')
wnlem = nltk.WordNetLemmatizer()
#Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language.

nltk.download('punkt')
#This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.

"""## Importing the dataset"""

dataset = pd.read_csv('Final_IPC_label_data.csv')  # This data contains the labelled definitions of IPC 302,307 and 376

dataset.head()     #The head() function is used to get the first n rows.

"""## Cleaning the texts"""

corpus = []     # defining a list of corpus
for i in range(0, 578):  # the loop for traversing through the rows
  definition = re.sub('[^a-zA-Z]', ' ', dataset['Definition'][i])     # the operation takes input of all word including alphabet
  definition = definition.lower()     # converts that into lower case (normalization and cleaning)
  definition = definition.split()   #split() method returns a list of strings after breaking the given string by the specified separator.
  wnlem = nltk.WordNetLemmatizer()  #brings context to the words.
  all_stopwords = stopwords.words('english') #useless words (data), are referred to as stop words.
  definition = [wnlem.lemmatize(word) for word in definition if not word in set(all_stopwords)]  # traversing through the words and normalizing it 
  definition = ' '.join(definition)     #Join all items in a tuple into a string, using a space (' ') character as separator:
  corpus.append(definition)            # filtered definition are added to the list

print(corpus)

"""## Creating the Bag of Words model"""

from sklearn.feature_extraction.text import CountVectorizer  #Convert a collection of text words to a matrix of token counts
cv = CountVectorizer( max_features = 620)
#With CountVectorizer we are converting raw text to a numerical vector representation of words. 
#This makes it easy to directly use this representation as features in Machine Learning tasks such as for text classification and clustering.
X = cv.fit_transform(corpus).toarray() #one step fit tranform
#Here the fit method, when applied to the training dataset,learns the model parameters (for example, mean and standard deviation). 
#We then need to apply the transform method on the training dataset to get the transformed (scaled) training dataset.
y = dataset.iloc[:, -1].values

len(X[0])

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""## Training the Support Vector Machine model on the Training set"""

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Saving our classifier
with open('C:/DEPLOYMENT/SVMclassifier.pkl','wb') as model_SVM_pkl:
    pickle.dump(classifier,model_SVM_pkl)
    
# Saving the BAG OF WORDS model
with open('C:/DEPLOYMENT/bagofwordsmodel.pkl','wb') as model_BOW_pkl:
    pickle.dump(cv,model_BOW_pkl)
    
