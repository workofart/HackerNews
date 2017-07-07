import numpy as np
import math
import json
import random
from nltk import word_tokenize, sent_tokenize
from dataAnalysis.dataset_utils import constructStory
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
from tqdm import tqdm
import gensim
import logging
import re
from gensim import corpora, models
from pymongo import MongoClient

##### DB Settings #####
client = MongoClient()
db = client['hn-db']
tags_collection = db.tags
items_collection = db.items

TRAIN_FILE = 'D:\Github\HackerNews\dataAnalysis\\training_mini.json'
DATA_LOCATION = 'D:\Github\HackerNews\dataAnalysis\working_data\\'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data = []
stop_words = stopwords.words('english')
translator = str.maketrans('', '', punctuation)
blacklist = ['rel', 'href', 'http', 'nofollow', 'nbsp', 'https', 'quot', 're', 've', 'gt']


def getTitles(data):
    """
    Extracts and tokenizes the title of a data item
    :param data: usually a story submission
    :return: tokenized title
    """
    if 'title' in data:
        return tokenize(data['title'])


def getCommentText(data):
    """
    Traverses the JSON tree and extracts the comment text
    from all the descendants recursively
    :param data: the parent node
    :return: list of tokenized comments
    """
    if 'kids' not in data and 'text' in data:
        return data['text']
    else:
        returnedText = []
        for kid in data['kids']:
            if type(kid) is int:
                print(data)
                data['kids'].remove(kid)
            else:
                if 'text' in data and data['text'] != '':
                    returnedText.append(data['text'])
                childComments = getCommentText(kid)
                if type(childComments) is list:
                    returnedText.extend(childComments)
                else:
                    returnedText.append(childComments)
            # if type(returnedText) is list:
            #     text.extend(returnedText)
            # else:
            #     print('here')
            #     text.append(returnedText)
        return returnedText


def getSamples(data, n):
    """
    Randomly selects "n" stories from the data array
    :param data: the parent array that samples will be selected from
    :param n: The number of samples to pick
    :return: array of selected samples
    """
    return [data[i] for i in sorted(random.sample(range(len(data)), n))]


def getFixedSamples(data):
    """
    Returns a list of list of data for testing purposes
    :param data: the parent list of all the json objects
    :return: list of json objects (dicts)
    """
    for item in data:
        if ('descendants' in item and item['descendants'] > 30):
            print(data.index(item))
    return [data[28], data[52], data[122]]


def getTermCount(term, doc):
    """
    TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
    Simple implementation of getting the term counts
    :param term:
    :param doc:
    :return: # of occurrence of term in doc
    """
    counter = Counter(doc)
    return counter[term]


def getTfidf(data):
    """
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
    Self implementation of TFIDF
    Exports to a file after completion
    Should be used by reading from tokens_[x].json OR After exportTokens()
    :param data: a parent list of json objects
    :return:
    """
    counterStorage = []
    finalStorage = []

    # For every document, create a counter to store term counts
    for item in data:
        doc = item['tokens']
        counter = Counter(doc)
        counterStorage.append(counter)
    # Second loop: collect number of documents with term t in it
    for item in tqdm(data):
        doc = item['tokens']
        uniqueTerms = set(doc)
        storage = []
        for term in uniqueTerms:
            docCount = 0
            for counter in counterStorage:
                if term in counter.keys():
                    docCount += 1
            temp = {}
            # There must be at least 1 docCount (itself)
            if docCount > 0:
                temp[term] = round(math.log(len(data) / docCount), 4)
            storage.append(temp)
        finalStorage.append({'id' : item['id'], 'tfidf' : storage})

    # Export to a file
    with open(DATA_LOCATION + 'tfidf_mini_' + str(len(data)) + '.json', 'w') as f:
        json.dump(finalStorage, f)


def tokenize(data):
    """
    Main utility tokenizer
    - removes stop words
    - removes 1 character
    - removes alphanumeric mixed junk
    - removes punctuation (both within string or by itself)
    - removes very long junk
    :param data: list of strings to tokenize
    :return: list of filtered tokens
    """
    cleanedTokens = []
    # Handle case of one string
    if type(data) is str:
        data = [data]

    for string in data:
        sentences = sent_tokenize(str(string))
        tokens = []
        for sent_token in [word_tokenize(sent) for sent in sentences]:
            tokens += sent_token

        cleanedTokens += [word.lower().translate(translator)
                          for word in tokens
                          if word.lower() not in stop_words
                          and word not in punctuation
                          and len(word.translate(translator)) > 1
                          # and re.search('^(?=.*[a-zA-Z])(?=.*[0-9])', word) is None
                          and re.search('^(?=.*[0-9])', word) is None
                          and len(word) > 1
                          and len(word) < 25
                          and word.lower().translate(translator) not in blacklist]
    return cleanedTokens


def exportTokens(data):
    """
    exports the list of tokens to a file for future usage
    Wraps the tokenizers
    :param data: the list of json objects to process
    :param n: the number of
    """
    export_data = []

    # try:
    for item in tqdm(data):
        print('Processing: ' + str(item['id']))
        allTokens = []
        if 'kids' in item:
            commentText = getCommentText(item)
            allTokens += tokenize(commentText)
        allTokens += getTitles(item)
        export_data.append({'id' : item['id'], 'tokens' : allTokens})

    with open(DATA_LOCATION + 'tokens_mini_' + str(len(data)) + '.json', 'w') as f:
        json.dump(export_data, f, ensure_ascii=True)


def importTokens(n):
    """
    Imports the tokenized stories
    Format:
    {
        "[id]": ["[word]"]
    }
    :param n: specifies which file to read from
    :return: the data from the file
    """
    with open(DATA_LOCATION + 'tokens_mini_' + str(n) + '.json', 'r') as f:
        data = json.load(f)
        return data


def tokenizeProcessor(n):
    """
    Mostly used by the Word2Vec pipeline
    :param n: the number of samples (defines which raw file to read)
    :return: list of tokenized docs
    :Note: the title and comments will be a separate record in the list
    """
    with open(DATA_LOCATION + 'training_mini_' + str(n) + '.json', 'r') as f:
        data = json.load(f)
        allWords = []
        for item in tqdm(data):
            if 'kids' in item:
                commentText = getCommentText(item)
                for comment in commentText:
                    if type(comment) is str:
                        allWords.append(tokenize(comment))
                    if type(comment) is list:
                        for sentence in comment:
                            allWords.append(tokenize(sentence))
            allWords.append(getTitles(item))
    return allWords


def word2VecVocabularyExport(n):
    """
    Build the vocabulary for the Word2Vec pipeline
    Saves the fully built vocabulary to a file
    :param n: the number of samples
    :return:
    """
    model = gensim.models.Word2Vec(size=400, iter=1, min_count=10, window=10, sample=0.001)
    allWords = tokenizeProcessor(n)
    model.build_vocab(allWords)
    model.save('model_vocabulary_' + str(n))


def trainWord2Vec(n, vocabulary_n):
    """
    Runs the tokenizer on the
    Loads the vocabulary file
    Trains the model using Word2Vec
    Exports the trained model to a file
    :param n: The number of samples
    :param vocabulary_n: The number of samples in the vocabulary
    """
    allWords = tokenizeProcessor(n)
    model = gensim.models.Word2Vec.load('model_vocabulary_' + str(vocabulary_n))
    model.train(allWords, epochs=model.iter, total_examples=model.corpus_count)
    model.save('model_' + str(vocabulary_n) + '_trained_' + str(n))


def testWord2Vec(n, vocabulary_n):
    """
    Loads the trained model from file, and performs some
    preliminary tests on it
    :param n: the target sample size (should be same as vocabulary n)
    :param vocabulary_n: only affects which file to read
    :return: the trained model
    """
    model = gensim.models.Word2Vec.load('model_' + str(vocabulary_n) + '_trained_' + str(n))
    print(model.wv['company'])
    print(model.similarity('apple', 'europe'))
    print(model.most_similar("startup"))
    return model
    # print(type(model.syn0_lockf))
    # print(model.syn0_lockf.shape)


def generateDocs(n):
    """
    Very similar to TokenizeProcessor, however, each story is a separate record in the returned list
    :param n: number of samples, defines which raw file to read
    :return: List of list of tokens
    """
    logging.info('Generating docs array')
    with open(DATA_LOCATION + 'training_mini_' + str(n) + '.json', 'r') as f:
        data = json.load(f)
        allWords = []
        ids = []
        for item in tqdm(data):
            ids.append(item['id'])
            itemWords = []
            if 'kids' in item:
                commentText = getCommentText(item)
                itemWords+= tokenize(commentText)
            itemWords += getTitles(item)
            allWords.append(itemWords)
    return allWords, ids


def kMeanTest(n, vocabulary_n):
    """
    Testing k-means clustering
    Using the Word2Vec word vectors
    Default: 25 clusters
    :param n: used for reading from file
    :param vocabulary_n: used for reading from file
    Prints out the most representative words of every cluster
    :return:
    """
    from sklearn.cluster import KMeans
    import time

    start = time.time()  # Start time
    model = testWord2Vec(n, vocabulary_n)
    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0.astype(np.float64)

    # It's a convention to have 5 words per cluster, but this is not our use case
    num_clusters = word_vectors.shape[0] / 5

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=25, verbose=True)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: ", elapsed, "seconds.")

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    # For the first 10 clusters
    for cluster in range(20):
        #
        # Print the cluster number
        print("\nCluster %d" % cluster)
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in range(len(word_centroid_map.values())):
            vals = list(word_centroid_map.values())
            keys = list(word_centroid_map.keys())
            if (vals[i] == cluster):
                words.append(keys[i])
        print(words)


def constructBagOfWordsVector(n, newN):
    """
    Pre-requisite: Have a training dictionary saved (run ldaTrain)
    Creates a list of docs for the test set
    Loads the dictionary from the training set - given "n"
    Creates the testing corpus relative to the dictionary tokens
    :param n: # of samples for the training set
    :param newN: # of samples in the testing set
    :return: corpus, and list of testing docs
    """

    # Assuming the list of docs are in the same order as the generated ids
    listOfDocs, ids = generateDocs(newN)
    logging.info("Creating the bag of words for testing set...")
    dictionary = corpora.Dictionary()
    dictionary = dictionary.load(DATA_LOCATION + 'LDA_' + str(n) + '_dictionary')
    corpus = [dictionary.doc2bow(text) for text in listOfDocs]

    # for item in corpus:
    #     for i in item:
    #         print("%s : [%s]" % (dictionary.get(i[0]), i[1]))
    #     print("\n\n")
    return corpus, listOfDocs, ids


def ldaTrain(n, num_topics):
    """
    Generates the dictionary for the training set
    Saves the dictionary to disk
    Trains the LDA model on the training set using multi-core CPUs
    # of workers = CPU cores - 1
    Saves the trained model to disk for future usage
    Prints the first 20 words in each cluster for reference
    :param n: the number samples in the training set
    :param num_topics: number of clusters/topics to train to
    """
    docs = generateDocs(n)
    logging.info('LDA Process Initialized')

    # create a Gensim dictionary from the texts
    dictionary = corpora.Dictionary(docs)

    # remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8, keep_n=None)
    dictionary.filter_n_most_frequent(30)

    dictionary.save('LDA_' + str(n) + '_dictionary')

    # convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in docs]

    lda = models.LdaMulticore(corpus, num_topics=num_topics,
                              id2word=dictionary,
                              eval_every=5,
                              chunksize=7000,
                              passes=100)

    lda.save('LDA_' + str(n) + '_model')
    topics_matrix = lda.show_topics(formatted=False, num_words=20, num_topics=num_topics)

    cid = 0
    for list in topics_matrix:
        sublist = list[1]
        print('Cluster %d' % cid)
        print([i[0] for i in sublist])
        cid += 1


def ldaTest(n, newN, num_topics):
    """
    Loads the trained LDA model from disk
    Prints the first 20 words in each cluster/topic
    Tests the model on the testing set
    Prints the test and training key words by topic for direct comparison
    :param n: # of samples in training set (used for specifying file name)
    :param newN: # of samples in testing set (used for specifying file name)
    :return:
    """
    model = gensim.models.LdaMulticore.load(DATA_LOCATION + 'LDA_' + str(n) + '_model')
    topics_matrix = model.show_topics(formatted=False, num_topics=num_topics, num_words=20)

    # cid = 0
    # for list in topics_matrix:
        #     sublist = list[1]
        # print('Cluster %d' % cid)
        # print([i[0] for i in sublist])
        # cid += 1

    # print('\n\n\n')
    vector, listOfDocs, ids = constructBagOfWordsVector(n, newN)

    # Load the tags file
    tags = []
    with open (DATA_LOCATION + 'LDA_' + str(n) + '_tags.txt', 'r') as f:
        tags = [line.strip() for line in f.readlines()]



    # plain bag-of-words count vectors
    doc_lda = model[vector]

    # Print out the predicted cluster and original document
    for cluster, doc, i in zip(doc_lda, listOfDocs, ids):
        print('---------------------')
        sorted_by_second = sorted(cluster, key=lambda tup: tup[1])
        tag_preds = sorted_by_second
        # print(target)
        print('Tokens: %s' % doc)
        print('Story Id: %d' % i)
        # for t in tag_preds:
        if len(tag_preds) > 2:
            tag_preds = tag_preds[-3:]
        for t in range(len(tag_preds)-1, -1, -1):
            sublist = topics_matrix[tag_preds[t][0]][1]
            # print('Cluster [%d] Key Words: %s' % (t[0], [i[0] for i in sublist]))
            print('Predicted Tag: %s' % tags[tag_preds[t][0]])


def ldaProcessOne(n, data, model, vector, id):

    # Load the tags file
    tags = []
    with open(DATA_LOCATION + 'LDA_' + str(n) + '_tags.txt', 'r') as f:
        tags = [line.strip() for line in f.readlines()]

    # plain bag-of-words count vectors
    doc_lda = model[vector]

    # Print out the predicted cluster and original document
    tagOutput = []
    for cluster in doc_lda:
        # print('---------------------')
        sorted_by_second = sorted(cluster, key=lambda tup: tup[1])
        tag_preds = sorted_by_second
        # print(target)
        # print('Tokens: %s' % doc)
        # print('Story Id: %d' % id)
        # for t in tag_preds:
        if len(tag_preds) > 2:
            tag_preds = tag_preds[-3:]
        for t in range(len(tag_preds) - 1, -1, -1):
            # print('Predicted Tag: %s' % tags[tag_preds[t][0]])
            tagOutput.append(tags[tag_preds[t][0]])
    # print(tagOutput)
    return tagOutput

def loadVocabulary(n):
    dictionary = corpora.Dictionary()
    dictionary = dictionary.load('LDA_' + str(n) + '_dictionary')
    print('ets')


def addLabelsToDB(n):
    dictionary = corpora.Dictionary()
    dictionary = dictionary.load(DATA_LOCATION + 'LDA_' + str(n) + '_dictionary')
    model = gensim.models.LdaMulticore.load(DATA_LOCATION + 'LDA_' + str(n) + '_model')

    # Process stories only
    for id in tqdm(range(0, 12862100)):
        data = items_collection.find_one({'id': int(id), "deleted" : { "$exists" : False }, "dead" : { "$exists" : False }})
        if (data is not None
            and data['type'] == 'story'
            and 'title' in data):

            id = data['id']
            itemWords = []
            if 'kids' in data:
                data = constructStory(int(id), {})
                commentText = getCommentText(data)
                itemWords.append(tokenize(commentText))
                itemWords[0].extend(getTitles(data))
            else:
                itemWords.append(getTitles(data))

            corpus = [dictionary.doc2bow(text) for text in itemWords]
            tag = ldaProcessOne(n, data, model, corpus, id)

            tags_collection.insert({"id" : id, "tag": tag})

if __name__ == '__main__':
    # ldaTrain(500000, 25)
    # ldaTest(500000, 200, 25)
    # loadVocabulary(500000)
    addLabelsToDB(500000)