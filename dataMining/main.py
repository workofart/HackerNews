import _pickle, json, requests

############# Parameters #############
fileName = 'PassiveIncome2016'
fileNameTest = 'Test'
item = 13150144
itemTest = 13142647

## Todo:
# Ranking of stories that are worth investigating
# - Decendents (number of comments in total for this story)
# - Score (complex derived number)

# Create a front-end web-page that can easily navgiate through the stories/asks
# - Sort functions
# - Summarizing functions
# - Categorizing functions
# - Sentiment functions


# Create an entry-point to understanding user behavior towards a certain topic/product
# - word frequency that relate to the dimensions of the product
    # E.g. For cars, handling would be one dimension, and the frequency of 'handling'
    # Can be used to describe the "importance" of that dimension




# 13139638 - Ask HN: What problem in your industry is a potential startup?

def getTopStories():
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty'
    itemSet = requests.get(url).json()
    i = 0
    for item in itemSet:
        i += 1
        print('Processing [' + str(i) + '] top story ' + str(item))
        with open(str(item) + '.json', 'w') as f:
            json.dump(crawlKids(item, 0), f, indent=4)


def getTopAsks():
    url = 'https://hacker-news.firebaseio.com/v0/askstories.json?print=pretty'
    itemSet = requests.get(url).json()
    i = 0
    for item in itemSet:
        i += 1
        print('Processing [' + str(i) + '] top asks: ' + str(item))
        with open(str(item) + '.json', 'w') as f:
            json.dump(crawlKids(item, 0), f, indent=4)


# Used to get comment text
def getText(jsonObj):
    return jsonObj.json()['text']


def getKids(jsonObj):
    if 'kids' in jsonObj.json():
        if jsonObj.json()['kids'] is None:
            return None
        else:
            return jsonObj.json()['kids']
    else:
        return None

def getParent(jsonObj):
    return jsonObj.json()['parent']

def getType(jsonObj):
    return jsonObj.json()['type']

def getUrl(item):
    return 'https://hacker-news.firebaseio.com/v0/item/' + str(item) + '.json?print=pretty'

def getResObj(item):
    return requests.get(getUrl(item))

def crawlKids(itemNumber, depth):

    # print('depth = ' + str(depth))
    url = getUrl(itemNumber)
    res = requests.get(url)

    kids = getKids(res)
    kidCount = 0
    storage = res.json()


    # Base case
    if (kids == None):
        # print('Reached base case at : ' + str(itemNumber) + ' depth: ' + str(depth))
        return res.json()
    else:
        kidStorage = []
        for kid in kids:
            # print('kids: ' + str(kids))
            # Store the comment
            res = getResObj(kid)

            # Make sure the kid has commnet element
            if res is None or res.json() is None:
                print('kid: ' + str(kid) + ' is None')
            elif 'type' in res.json() and getType(res) == 'comment':
                    kidStorage.append(crawlKids(kid, depth+1))

        storage['kids'] = kidStorage
        return storage

    # print('Parent type: ' + getType(getResObj(getParent(res))))
    # print(kids)


def getOneStory(item):
    with open(fileName + '.json', 'w') as f:
        json.dump(crawlKids(item, 0), f, indent=4)


# getTopAsks()
getOneStory(item)
getTopStories()