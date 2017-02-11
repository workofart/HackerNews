import pprint, ujson, bson, pickle
from pymongo import MongoClient

def exportDataset(list, fileName):
    path = 'corpus/' + fileName

    with open(path, 'wb') as outFile:
        pickle.dump(list, outFile)

def importDataset(fileName):
    path = 'corpus/' + fileName
    with open(path, 'rb') as inFile:
        print(pickle.load(inFile))

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


client = MongoClient()
db = client['hn-db']
collection = db.items
fileName = 'temp.json'
titleList = []
textList = []
commentList = []
magicNumber = 300

for x in range(1, magicNumber):
    lowerLimit = x * (magicNumber - 1) - (magicNumber - 2)
    upperLimit = x * (magicNumber - 1)

    # pprint.pprint()

    # extract title and text content
    items = collection.find({'type': 'story', 'id': {'$gte': lowerLimit, '$lte': upperLimit}})
    for item in items:
        # pprint.pprint(item)
        if ('title' in item):
            titleList.append(item['title'])
        if ('text' in item):
            textList.append(item['text'])

        # query the kids
        commentList = getKids(item, commentList, collection)
        print('Finished one story')
# print('-------------- Title List -------------')
# pprint.pprint(titleList)
# print()
# print('-------------- Story Text List -------------')
# pprint.pprint(textList)
# print()
# print('-------------- Comment List -------------')
# pprint.pprint(commentList)
# print()

# exportDataset(titleList, 'titleList')
# exportDataset(commentList, 'commentList')
# exportDataset(textList, 'textList')

# importDataset('commentList')
