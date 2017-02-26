import pprint, ujson, bson, pickle, nltk, html, math
import pandas as pd
from pymongo import MongoClient
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

##### DB Settings #####
client = MongoClient()
db = client['hn-db']
collection = db.items
vocabulary = db.vocabulary # top 10 most common
vocabulary_all = db.vocabulary_compressed
#######################

def bagOfWords(training_set):
    vectorizer = CountVectorizer(analyzer="word",   \
                                 tokenizer=None,    \
                                 preprocessor=None, \
                                 stop_words=None,   \
                                 max_features=5000)
    train_data_features = vectorizer.fit_transform(training_set)
    train_data_features = train_data_features.toarray()
    print()
    print(u'Shape of features: {}'.format(train_data_features.shape))
    # vocab = vectorizer.get_feature_names()
    # print(vocab)
    # dist = np.sum(train_data_features, axis=0)
    # for tag, count in zip(vocab, dist):
    #     print(count, tag)
    # print(u'Count of each vocabulary: {}'.format())

def randomForest(training_set_features):
    forest = RandomForestClassifier(n_estimators=100)

    forest = forest.fit(training_set_features, train["sentiment"])

def prettyPrintWordFreq(dictSet):
    for word, freq in dictSet:
        print(u'{};{}'.format(word, freq))


def getKids(node, commentList, collection):
    # check if the current node has text and only print if it has depth of 1
    if node is not None and 'kids' in node:
        kids = node['kids']
        if (kids is not None and len(kids) > 0):
            for kid in kids:
                kidObj = collection.find_one({'id': kid})
                getKids(kidObj, commentList, collection)
    if node is not None and 'text' in node and 'dead' not in node:
        comment = node['text']
        commentList.append(comment)
    return commentList

def statusUpdate(interval, i, total):
    if (i % interval == 0):
        progress = (i / total) * 100
        print ("\n \n Processing Story %d of %d (%.2f)%% \n \n" % (i, total, progress))

def processComments(comments):
    aggregateComments = []
    for comment in comments:
        # print('====== Before =========')
        # print(comment)
        stop = stopwords.words('english')
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        comment = list((x.encode('utf-8') for x in word_tokenize(html.unescape(comment.lower().translate(translator))) if x not in stop))
        # print('====== After =========')
        # print(comment)
        aggregateComments.extend(comment)
        # aggregateComments = list(set(aggregateComments))
    fdist = FreqDist(w for w in aggregateComments)
    return fdist.most_common()


# Checks if the story was inserted or not
def checkVocabExist(id):
    items = vocabulary_all.find({'id' : id})
    if (items.count() != 0):
        return True
    else:
        return False

# Process a given range of stories:
# 1. print out the vocabulary freq of the comments
# 2. Print out the vocabulary freq of the title words
# Return: a list of comments from all the processed stories
def processStories(lowerBound, upperBound, dbFlag, fileFlag, aggregatedCommentsFlag=0, partitions=5):
    # temp storage
    titleList = []
    textList = []
    commentList = []
    aggregatedComments = []
    # csvArray = []
    title = ''
    items = collection.find({'type': 'story', 'id': {'$gte': lowerBound, '$lte': upperBound}, 'descendants' : {'$gt' : 0}}, no_cursor_timeout=True)
    totalItems = items.count()
    print("There are %d matched stories" % totalItems)
    stepSize = math.ceil(totalItems / partitions)
    i = 1
    for item in items:
        statusUpdate(stepSize, i, totalItems)
        # print('Processing id: %s' % item['id'])
        if (fileFlag is True or checkVocabExist(item['id']) is False):
            # print('Id: %s wasn\'t analyzed yet' % item['id'])
            if ('title' in item):
                titleList.append(item['title'])
            if ('text' in item):
                textList.append(item['text'])
            commentList = getKids(item, commentList, collection)
            if (aggregatedCommentsFlag):
                aggregatedComments.extend(commentList) # adds the comment of one story to the list of comments
            if (len(commentList) != 0):
                # print(processComments(commentList))

                # insert the story with associated commments, id, and title
                # Process comment tuples
                processedComments = processComments(commentList)
                objList = []
                csvArray = []
                for comment in processedComments:
                    word = comment[0]
                    freq = comment[1]
                    if (dbFlag is True):
                        entry = {"word": word, "freq": freq}
                        objList.append(entry)
                    if (fileFlag is True):
                        csvEntry = [str(word, 'utf-8'), freq, item['id']]
                        csvArray.append(csvEntry)
                if(dbFlag is True):
                    # Insert the constructed object back into the db
                    insertedItem = vocabulary_all.insert_one({"title" : item['title'], "vocabulary" : objList, "id" : item['id']})
                if (fileFlag is True):
                    dataSet = pd.DataFrame(data=csvArray, columns=['Word', 'Freq', 'Id'])
                    # Insert the object into an external file
                    try:
                        with open('test.csv', 'a') as f:
                            dataSet.to_csv(f, index=False, header=False)
                    except UnicodeEncodeError:
                        print('Cannot encode string: ' + str(i))
        else:
            print('Id: %s was analyzed already' % item['id'])
        commentList = []
        i += 1
    items.close() # free up the resources
    # if (fileFlag is True):
    #     dataSet = pd.DataFrame(data=csvArray, columns=['Word', 'Freq', 'Id'])
    #     # Insert the object into an external file
    #     try:
    #         with open('test.csv', 'a') as f:
    #             dataSet.to_csv(f, index=False, header=False)
    #     except UnicodeEncodeError:
    #         print('Cannot encode string: ' + str(i))
    return aggregatedComments

def getVocabulary():
    items = vocabulary.find()
    for item in items:
        print(item)


# i = 253

# compressed: i = 281

for i in range(100,130):
    print("====> i: %d" % i)
    processStories(100000* (i - 1) + 1, 100000 * i, partitions=100, dbFlag=False, fileFlag=True)



# for i in range(1,100):
#     print("====> i: %d" % i)
#     processStories(10000* (i - 1) + 1, 10000 * i, partitions=100, dbFlag=False, fileFlag=True)

# processStories(8532250, 8532280, partitions=1, dbFlag=False, fileFlag=True)

########## >>>>> 3491542

# for i in range (1, 120):
#     print(i)
# getVocabulary()
# bagOfWords(processStories(0, 300))

# print(string.punctuation)

# print('Titles:')
# pprint.pprint(titleList)


# print('Text:')
# pprint.pprint(textList)

