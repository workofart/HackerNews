import _pickle, json, pprint, requests, numpy, scipy, nltk, path, pathlib

from nltk.corpus import brown

def getWordFreq(story):
    wordFreq = []
    return wordFreq


def rankByScores(stories):
    stories = sorted(stories, key='score')
    return stories

# Read the list of files and does processing on each file read
def readFiles(dirName):

    # get file list from directory

    p = pathlib.Path(dirName)
    files = p.iterdir()

    parentStories = []
    for file in files:
        with open(str(file), 'r') as f:
            data = json.load(f)

            # Extract the parent story/ask and store them into a list
            # parentStory = extractParentStory(data)
            buildCorpus(data);
        # parentStories.append(parentStory)
    # sortedScore_parentStories = sortByX(parentStories, 'score')

    # Get the top N stories by a certain dimension
    # pprint.pprint(getTopNStories(sortedScore_parentStories, 1))

# Function that will sort a list of custom objects by the key 'X'
def sortByX(list, X):
    return sorted(list, key=lambda k: k[X], reverse=True)

def extractParentStory(data):
    story = {}
    if 'title' in data:
        story['title'] = data['title']
    if 'score' in data:
        story['score'] = data['score']
    if 'descendants' in data:
        story['descendants'] = data['descendants']
    if 'id' in data:
        story['id'] = data['id']
    # if 'by' in data:
        # story['by'] = data['by']
    # if 'time' in data:
    #     story['time'] = data['time']
    # if 'type' in data:
    #     story['type'] = data['type']
    # if 'text' in data:
        # story['text'] = data['text']
    if 'url' in data:
        story['url'] = data['url']
    return story

def getTopNStories(list, n):
    return list[:n]

def buildCorpus(data):
    title = data['title'];

    # traverse the tree and keep the text of each comment

    print(data['title']);

def getCommentText(node):
    # check
    if (node['kids'].length > 0):

    node =

def movie_features():
    print(brown.categories())
    selectedCategories = ['news', 'fiction', 'reviews'];
    modals = ['startup', 'technology']
    cfd = nltk.ConditionalFreqDist((genre, word)
                                   for genre in brown.categories()
                                   for word in brown.words(categories=genre))
    cfd.tabulate(conditions=selectedCategories, samples=modals)


item = '13123136'
readFiles('../dataMining/top_asks')

# movie_features()



