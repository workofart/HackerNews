import _pickle, json, pprint, requests, numpy, scipy, nltk, path, pathlib, html, os

from nltk.corpus import brown

# Global Vars


def getWordFreq(story):
    wordFreq = []
    return wordFreq


def htmlPrint(text):
    print(html.unescape(text))

def rankByScores(stories):
    stories = sorted(stories, key='score')
    return stories

# Read the list of files and does processing on each file read
def readFiles(dirName):
    n = 10  # top comments with more than N kids

    # get file list from directory
    p = pathlib.Path(dirName)
    files = p.iterdir()

    parentStories = []
    for root, subdirs, files in os.walk(dirName):
        parentRoot = root
        print("Root: " + root)

        for subdir in subdirs:
            print('\t- subdir ' + subdir)

        for filename in files:
            topComments = []
            file_path = os.path.join(root, filename)

            outFile = os.path.join('..\\dataMining\\data', 'topComments', filename.rstrip('.json') + '.json')
            # print('\t\t- file %s \n\t\t- outputFile %s' % (file_path, outFile))

            with open(file_path) as f:
                # data = []
                # for line in f:
                #     data.append(json.load(line))
                # print(dir(f));
                try:
                    data = json.load(f)
                except ValueError:
                    print('Decoding error for ' + f.name)
                # Extract the parent story/ask and store them into a list
                parentStories.append(extractParentStory(data))
                getTopComments(data, n, topComments)

                print('There are %s top comments for the file %s' % (len(topComments), f.name))
                with open(outFile, 'w') as out:
                    json.dump(topComments, out, indent=4)

                buildCorpus(data, topComments);
    # sortedScore_parentStories = sortByX(parentStories, 'score')

    # Get the top N stories by a certain dimension
    # pprint.pprint(getTopNStories(sortedScore_parentStories, 1))

def readFile(file):
    with open(str(file)+'.json', 'r') as f:
        data = json.load(f)
        topComments = getTopComments(data)
        print(topComments)

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
    if 'url' in data:
        story['url'] = data['url']
    # if 'by' in data:
        # story['by'] = data['by']
    # if 'time' in data:
    #     story['time'] = data['time']
    # if 'type' in data:
    #     story['type'] = data['type']
    # if 'text' in data:
        # story['text'] = data['text']
    return story

def getTopNStories(list, n):
    return list[:n]

# Filter through all the first level comments that don't have kids
# Return a list of nodes that don't contain 'useless' comments
def cleanNoKidComments(data):
    if 'kids' in data:
        kids = data['kids']
        # loop through all the kids
        for kid in kids:
            # if 'kids' in kid and len(kid['kids']) == 0:
            if 'kids' in kid and len(kid['kids']) == 1:
                print(kid['kids'])
                # htmlPrint(kid['text'])
    return data


# Each comment has its own children, and by counting them we can see
# the popularity of each comment, and apply sort/filter
def getTopComments(data, n, topComments):
    # reached the leaf child
    if 'kids' not in data:
        # data['decendents'] = 0
        return 0
    else:
        # reset the decendents for the current node
        kids = data['kids']
        if (kids is not None and len(kids) > 0):
            decendents = 0
            for kid in kids:
                kid['decendents'] = getTopComments(kid, n, topComments)
                if kid['decendents'] != None:
                    decendents = decendents + kid['decendents'] + 1

                    # Print out the comments with children > n
                    if (kid['decendents'] > n):
                        tmpObj = {}
                        tmpObj['id'] = kid['id']
                        tmpObj['text'] = html.unescape(str(kid['text']))
                        tmpObj['decendents'] = kid['decendents']
                        topComments.append(tmpObj)
            if (kid['type'] == 'story'):
                print()
                print('Story children: ' + htmlPrint(str(kid['decendents'])))
            return decendents



def buildCorpus(data):
    title = data['title'];

    print('=========== ' + title + '===========');

    # cleanNoKidComments(data);
    # traverse the tree and keep the text of each comment
    getCommentText(data, 0)


def getCommentText(node, depth):

    # check if the current node has text and only print if it has depth of 1
    if 'text' in node:
        comment = node['text']
        tabs = ''
        for x in range(0, depth):
            tabs += '\t'
        comment = tabs + comment
        print()
        print('Comment: ' + htmlPrint(comment))

    if 'kids' in node:
        kids = node['kids']
        if (kids is not None and len(kids) > 0):
            for kid in kids:
                getCommentText(kid, depth + 1)
    # reached the leaf child
    else:
        print()

def movie_features():
    print(brown.categories())
    selectedCategories = ['news', 'fiction', 'reviews'];
    modals = ['startup', 'technology']
    cfd = nltk.ConditionalFreqDist((genre, word)
                                   for genre in brown.categories()
                                   for word in brown.words(categories=genre))
    cfd.tabulate(conditions=selectedCategories, samples=modals)


item = '13123136'
readFiles('..\\dataMining\\data')
# readFile('../dataMining/data/top_stories_20170101/13274871')
# nltk.download()
# movie_features()




# dataFrame = pd.read_csv('test-000.csv', encoding='ISO-8859-1')
# print(dataFrame.dtypes)

# del dataFrame

# for i in range(1,10):
#     print(i)
#     dataFrame = pd.read_csv('test-00' + str(i) + '.csv', encoding='ISO-8859-1', error_bad_lines=False)
#     dataFrame.to_hdf('vocabulary.h5', 'vocabulary', mode='w', format='table')
#     del dataFrame

# for i in range(11,100):
#     print(i)
#     dataFrame = pd.read_csv('test-0' + str(i) + '.csv', encoding='ISO-8859-1', error_bad_lines=False)
#     dataFrame.to_hdf('vocabulary.h5', 'vocabulary', mode='a', format='table')
#     del dataFrame

# for i in range(100,153):
#     print(i)
#     dataFrame = pd.read_csv('test-' + str(i) + '.csv', encoding='ISO-8859-1', error_bad_lines=False)
    # print(dataFrame[dataFrame['id'] == 605055])
    # dataFrame = dataFrame.drop(dataFrame[dataFrame['id'] == 605055].index)
    # dataFrame.to_csv('test-00' + str(i) + '.csv', encoding='ISO-8859-1')
    # if (dataFrame[dataFrame['id']] > 0 and dataFrame[dataFrame['id']] < 1000000):
    # if (len(dataFrame[].index) != 0):
    #     print(dataFrame[dataFrame['word']])
    #     print(dataFrame[dataFrame['id'] == 605055])
    # dataFrame.to_hdf('vocabulary.h5', 'vocabulary', mode='a', format='table')
    # del dataFrame
# print('----------------')
# print(pd.read_hdf('vocabulary.h5', 'vocabulary'))



#### Tokenizing text ####
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(train_text)
# print(X_train_counts.shape)

#### Use Term Frequency times Inverse Document Frequency ####
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)


#### Training a classifier ####
# classifier = MultinomialNB().fit(X_train_tfidf, train_label)

# subTest_text = test_text[0:10]
# subTest_label = test_label_text[0:10]

# X_test_counts = count_vect.transform(subTest_text)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# predicted = classifier.predict(X_test_tfidf)

