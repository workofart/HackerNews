import numpy as np
from numpy import isnan
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import json, csv
from json.decoder import JSONDecodeError
dirPath = 'D:\\Github\\HackerNews\\dataAnalysis\\data\\'
filePath = dirPath + 'RC_2011-01'
subredditsFile = dirPath + 'subReddits.csv'
textFile = dirPath + 'fullText.csv'

def statusUpdate(interval, i, total):
    if (i % interval == 0):
        progress = (i / total) * 100
        print ("\n \n Processing Story %d of %d (%.2f)%% \n \n" % (i, total, progress))

def exportSubReddits():
    subreddits = dict()
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

    with open(csvFile, 'w') as f:
        for w in sortedSubreddits:
            f.write(w + ',' + str(subreddits[w]) + '\n')

def getSubReddit():
    # Read from csv to get list of subreddits
    with open(subredditsFile, 'r') as f:
        reader = csv.reader(f)
        temp = []
        for row in reader:
            temp.append(row[0])
    return temp

def getText(interval):
    subreddits = getSubReddit()
    print('Subreddits: \n', subreddits)
    storage = dict()
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
    # print(json.dumps(storage))

    with open(textFile, 'w') as f:
        # json.dump(storage, f)
        for key in storage.keys():
            for text in storage[key]:
                text = text.replace('\n', ' ').replace(',', ' ').replace('\r', ' ')
                try:
                    f.write(text + ',' + key + '\n')
                    # print('key: ', key)
                    # print('val: ', text)
                except UnicodeEncodeError:
                    print('Can\'t encode')


def classify(input):
    with open(textFile, 'r') as f:
        classifier = NaiveBayesClassifier(f, format='csv')
    return classifier.classify(input)

# Randomly shuffles the dataset and pick out 30% as training set, 30% as testing set
def preprocess():
    df = pd.read_csv(textFile, encoding='ISO-8859-1')
    print(df.shape)
    df.columns = ['text', 'label']
    df = df.astype('U')
    train, test = train_test_split(df, train_size=0.7, test_size=0.3)

    train_text = [f for f in train['text']]
    test_text = [f for f in test['text']]

    le = preprocessing.LabelEncoder()
    le.fit(train['label'])
    train_label = le.transform(train['label'])

    le.fit(test['label'])
    test_label = le.transform(test['label'])
    # train = [tuple(x) for x in train[['text', 'label']].values]
    # test = [tuple(x) for x in test[['text', 'label']].values]
    return train_text, train_label, test_text, test_label


# getText(10000)
# print(classify('This sentence contains guns'))
train_text, train_label, test_text, test_label = preprocess()

print('Size of training set: ', len(train_text))


print('Size of testing set: ', len(test_text))


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(train_text, train_label)


predicted = text_clf.predict(test_text)
print(np.mean(predicted == test_label))









