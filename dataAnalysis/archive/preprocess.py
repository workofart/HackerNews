# coding: utf-8
# This module is a utility module for supporting analysis.py

##### Standard Imports ######
import json, csv, itertools
from time import time
from os.path import join
from os import listdir
from json.decoder import JSONDecodeError

######　Scikit Learn ######
from sklearn.model_selection import train_test_split


###### Numpy ######
import numpy as np
from numpy import isnan

###### Pandas ######
import pandas as pd

###### MatPlotLib ######
import matplotlib.pyplot as plt

###### Textblob ######
from textblob.classifiers import NaiveBayesClassifier


###### FilePaths ######
dirPath = 'D:\\Github\\HackerNews\\dataAnalysis\\data'
files = ['RC_2011-01','RC_2010-12', 'RC_2012-01', 'RC_2012-03']
filePaths = [join(dirPath, f) for f in files]
subredditsFile = join(dirPath, 'subReddits.csv')
textFile = join(dirPath, 'fullText.csv')


def statusUpdate(interval, i, total):
    """
    Helper function used as a status updater/logger
    :param interval: how frequent for a given update
    :param i: input as the current counter, where 0 <= i <= total
    :param total: The total known items to be processed
    """
    if (i % interval == 0):
        progress = (i / total) * 100
        print ("\n \n Processing Story %d of %d (%.2f)%% \n \n" % (i, total, progress))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def exportSubReddits():
    """
    Utility function that loops through all comments and stores the
    subreddits count and subreddit title into a csv file
    to prepare for shortlisting the subreddits for sample construction
    :return:
    """
    subreddits = dict()
    for filePath in filePaths:
        with open(filePath, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except JSONDecodeError:
                    print('Error reading: ' + line)
                subreddit = obj['subreddit']
                text = obj['body']
                # blob = TextBlob(text)
                # for x in blob.words:
                #     print(x)
                if subreddit in subreddits:
                    subreddits[subreddit] += 1
                else:
                    subreddits[subreddit] = 1

        sortedSubreddits = sorted(subreddits, key=subreddits.get, reverse=True)
        # for w in sortedSubreddits[0:30]:
        #     print(w, subreddits[w])

    with open(subredditsFile, 'w') as f:
        for w in sortedSubreddits:
            f.write(w + ',' + str(subreddits[w]) + '\n')


def getSubReddit():
    """
    Utility function for reading subreddits array from csv file
    The csv file should be all the subreddits that you defined to be usable
    :return: list of subreddits from the csv
    """
    # Read from csv to get list of subreddits
    with open(subredditsFile, 'r') as f:
        reader = csv.reader(f)
        temp = []
        for row in reader:
            temp.append(row[0])
    return temp


def getText(interval):
    """
    RUN THIS ONLY IF SUBREDDITS HAVE CHANGED, if not, then all the text would be
    saved in separate csv files in the 'category' folder
    Given "subreddits" file, loop through all comments and save the ones
    in the given list
    :param interval: solely for status updating purposes
    :return:
    """
    subreddits = getSubReddit()
    print('Subreddits: \n', subreddits)
    print('There are %d subreddits in total' % len(subreddits))
    storage = dict()
    for filePath in filePaths:
        with open(filePath, 'r') as f:
            # totalLines = sum(1 for row in f)
            # f.seek(0)
            # i = 0
            for line in f:
                # statusUpdate(interval, i, totalLines)
                # i += 1
                try:
                    obj = json.loads(line)
                except JSONDecodeError:
                    print('Error reading: ' + line)
                subreddit = obj['subreddit']
                text = obj['body']
                # if item is in the selected subreddits, store the object
                if subreddit in subreddits:
                    if subreddit not in storage:
                        print('Creating key slot: ', subreddit)
                        storage[subreddit] = []
                    if text != '[deleted]':
                        storage[subreddit].append(text)

    for subreddit in subreddits:
        with open(join(dirPath, 'category', subreddit + '.csv'), 'a') as f:
            print('Writing to:', subreddit)
            for text in storage[subreddit]:
                text = text.replace('\n', ' ').replace(',', ' ').replace('\r', ' ')
                try:
                    f.write(text + '\n')
                except UnicodeEncodeError:
                    pass


def textBlobClassify(input):
    """
    Classify a sentence by using the TextBlob - Naive Bayes classifier
    (Very slow)
    :param input: Input text
    :return: returns the trained classifier
    """
    with open(textFile, 'r') as f:
        classifier = NaiveBayesClassifier(f, format='csv')
    return classifier.classify(input)


def preprocess(trainingFraction, testingFraction, minSample=2107):
    """
    Randomly shuffles the dataset and pick out predefined fraction % as training/testing set
    :param trainingFraction: in terms of % of total dataSet
    :param testingFraction:  in terms of % of total dataSet
    :param minSample: minimum number of samples from each category (subreddit)
    :return: training/testing set raw text and labels, selected subreddits
    """
    # Loop through all the files in the category folder
    categoryDir = join(dirPath, 'category')

    # Array for holding samples from all categories
    subreddits = []
    pos = 0
    fullDF = pd.DataFrame()
    for file in listdir(categoryDir):
        subreddits.append(file.replace('.csv', ''))
        csvFile = join(categoryDir, file)
        df = pd.read_csv(csvFile, encoding='ISO-8859-1').sample(n=minSample)
        df.columns = ['text']
        df = df.astype('U') # remove NaN encoding error
        df['label'] = pd.Series(np.ones(minSample) * pos, index=df.index)

        fullDF = fullDF.append(df)
        pos += 1
    print('Training set shape: ', str(fullDF.shape))

    # After all samples are constructed, create training/testing set
    train, test = train_test_split(fullDF, train_size=trainingFraction, test_size=testingFraction)
    train_text = [f for f in train['text']]
    test_text = [f for f in test['text']]

    train_label = [f for f in train['label']]
    test_label = [f for f in test['label']]
    # print('Total training size: %d' % len(train_full_text))
    # print('Total training size: %d' % len(train_full_label))
    # print('Total testing size: %d' % len(test_full_text))
    print('Subreddits: ', subreddits )
    return train_text, train_label, test_text, test_label, subreddits



def getTopNAccuracy(test_label, predicted_prob, n=3):
    """
    Getting the best N predictions
    :param test_label: the ground truth labels
    :param predicted_prob: the predicted labels
    :param n: N best predictions
    """
    print('------------------------------------------------')
    print('Getting best [', n, '] accuracy')
    total = len(predicted_prob)
    bestn = [(-np.array(sample)).argsort()[:n] for sample in predicted_prob]

    count = 0
    for i in range(0, total):
        # Printing out the best N labels
        # print('bestn: ' + str(bestn[i]) + ' | truth: ' + str(test_label[i]))
        if test_label[i] in bestn[i]:
            count += 1
    print('Count: ', count)
    print('Accuracy: ', count / total)


def getMinSamples():
    """
    RUN THIS AFTER RUNNING getText()
    Gets the minimum samples size across all subreddits to support
    Training/testing set construction
    :return: the minimum samples across all subreddits
    """
    minSamples = 9999999999
    categoryDir = join(dirPath, 'category')
    for file in listdir(categoryDir):
        csvFile = join(categoryDir, file)
        df = pd.read_csv(csvFile, encoding='ISO-8859-1')
        print('%s %d' % (file, df.shape[0]))


        # record the smallest dataset
        if minSamples > df.shape[0]:
            minSamples = df.shape[0]
    print('\n=======\nMin samples: ', minSamples)
    return minSamples


# RUN THIS ONLY IF DATASOURCE HAVE CHANGED
# exportSubReddits()

# RUN THIS ONLY IF SUBREDDITS HAVE CHANGED
# getText(1)
