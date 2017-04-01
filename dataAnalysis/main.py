# This module is used for extracting features from reddit posts and comments
# As a way to learn how to categorize a given post to a range of subreddits
# This learning was intended to be transferred for tagging HackerNews posts


##### Standard Imports ######
from time import time
import string, nltk

from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

######　Scikit Learn ######
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

###### Numpy ######
import numpy as np
np.set_printoptions(precision=3)
###### Pandas ######
import pandas as pd

###### MatPlotLib ######
import matplotlib.pyplot as plt

###### Requires Preprocess module ######
from dataAnalysis.preprocess import preprocess, getTopNAccuracy

stemmer = EnglishStemmer()
def stem_tokens(tokens, stemmer2):
    """
    Helper class to support stemming tokens
    :param tokens: the tokens to be stemmed
    :param stemmer2: the rules on how to stem tokens
    :return: the resulting array of stemmed tokens
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer2.stem(item))
    return stemmed


def tokenize(text):
    """
    Utility class for tokenizing and stemming words in the given text
    :param text: input text for tokenization
    :return: stemmed tokens generated from the given text
    """
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# Total
# minSample = 57337
train_text, train_label, test_text, test_label, subreddits = preprocess(0.7, 0.3, 57337)

# print('Shape of training set: ', np.array(train_text).shape)
# print('Shape of testing set: ', np.array(test_text).shape)
train_size = len(train_text)
test_size = len(test_text)

def NaiveBayesPipline():
    """
    Naive Bayes Classifier Pipeline
    :parameter:
        ngram_range in the CountVectorizer
    """
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3), tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    t0 = time()
    text_clf = text_clf.fit(train_text, train_label)

    ########## Taking a peek at the trained features ############
    # CountVectorizer_PARAMS = text_clf.get_params()['vect']
    # CountVectorizer_FEATURES = CountVectorizer_PARAMS.get_feature_names()
    # print('Length of features: ', len(CountVectorizer_FEATURES))
    # print(CountVectorizer_PARAMS)
    # print(CountVectorizer_FEATURES)

    ###### Top 1 ######
    # predicted = text_clf.predict(test_text)
    # print(np.mean(predicted == test_label))

    train_time = time() - t0
    print('Training time: ', train_time)
    print('Done training Naive Bayes Classifier')
    predicted_prob = text_clf.predict_proba(test_text)
    getTopNAccuracy(test_label, predicted_prob, 3)


def SVMPipeline():
    """
    Support Vector Machine Pipeline
    :parameter:
        ngram_range in the CountVectorizer
        n_iter in the SGDClassifier
    """
    text_clf_SVM = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3), tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(penalty='l2', alpha=0.0001, n_iter=np.ceil(10 ** 6 / train_size),
                                                   random_state=42, loss='log'))])
    t0 = time()
    text_clf_SVM = text_clf_SVM.fit(train_text, train_label)

    ###### Top 1 ######
    # predicted = text_clf_SVM.predict(test_text)
    # print(np.mean(predicted == test_label))

    train_time = time() - t0
    print('Training time: ', train_time)
    print('Done training SVM Classifier')
    predicted_prob_SVM = text_clf_SVM.predict_proba(test_text)
    getTopNAccuracy(test_label, predicted_prob_SVM, 3)






