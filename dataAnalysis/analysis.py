import pprint, ujson, bson, pickle, nltk, html, math
from pymongo import MongoClient
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

##### DB Settings #####
client = MongoClient()
db = client['hn-db']
collection = db.items
#######################

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
        print ("\n ===================== \n Story %d of %d \n ===================== \n" % (i, total))

def processComments(comments):
    aggregateComments = []
    for comment in comments:
        # print('====== Before =========')
        # print(comment)
        stop = stopwords.words('english')
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        comment = list((x for x in word_tokenize(html.unescape(comment.lower().translate(translator))) if x not in stop))
        # print('====== After =========')
        # print(comment)
        aggregateComments.extend(comment)
        # aggregateComments = list(set(aggregateComments))
    fdist = FreqDist(w for w in aggregateComments)
    return fdist.most_common(10)


def processStories(lowerBound, upperBound):
    # temp storage
    titleList = []
    textList = []
    commentList = []

    items = collection.find({'type': 'story', 'id': {'$gte': lowerBound, '$lte': upperBound}, 'descendants' : {'$gt' : 0}})
    totalItems = items.count()
    print("There are %d matched stories" % totalItems)
    stepSize = math.floor(totalItems / 5)
    i = 1
    for item in items:
        statusUpdate(stepSize, i, totalItems)
        if ('title' in item):
            titleList.append(item['title'])
            # print('Title: ' + item['title'])
        if ('text' in item):
            textList.append(item['text'])
        commentList = getKids(item, commentList, collection)
        if (len(commentList) != 0):
            print(processComments(commentList))
        # print('Comments: ' + str(processComments(commentList)))
        commentList = []
        i += 1

processStories(0, 1000)

# print(string.punctuation)

# print('Titles:')
# pprint.pprint(titleList)


# print('Text:')
# pprint.pprint(textList)

