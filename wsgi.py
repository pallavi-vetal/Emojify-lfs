import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from flask import Flask, flash, redirect, render_template, request, session, abort
lemmatizer = WordNetLemmatizer()
#from emo_utils import *
import emoji
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import load_model
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
app = Flask(__name__,static_url_path='/static')
import requests
import urllib2  # the lib that handles the url stuff
from zipfile import ZipFile 
from tqdm import tqdm
import requests


import requests, zipfile, io
url = "http://nlp.stanford.edu/data/glove.6B.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()  


def clean_sentences(df):
    reviews = [ ]

    for sent in df:
        
        #remove html content
        review_text = sent
        s = TextBlob(review_text)
        review_text = str(s.correct())
        #print(s)
        #remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
        #tokenize the sentences
        words = word_tokenize(review_text.lower())
        
        # stemming of words

        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        #lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]
    
        reviews.append(lemma_words)

    return(reviews)

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = len(X)                               
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = [ word.lower() for word in X[i]]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j+1
            
    ### END CODE HERE ###
    
    return X_indices
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

global graph
graph = tf.get_default_graph()
loaded_model = load_model('deploy-new.h5')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
print(word_to_index.len)
@app.route('/')
def hello_world():
   
    #loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return render_template('index.html')

@app.route('/emojify',methods = ['POST', 'GET'])
def emojify():
    user = request.form['emojify']
    print(user)
    x_test = np.asarray([user])
    x_test = clean_sentences(x_test)
    X_test_indices = sentences_to_indices(x_test, word_to_index, 10)
    s=''
    with graph.as_default():
            s = str(x_test[0]) +' '+  label_to_emoji(np.argmax(loaded_model.predict(X_test_indices)))
    
    return render_template('index.html',emoji=s)


