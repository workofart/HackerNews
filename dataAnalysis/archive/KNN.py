# coding: utf-8

# 1214 = 20 mins

import scipy, sklearn, string, nltk, logging, time
import pprint

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
from pymongo import MongoClient

##### DB Settings #####
client = MongoClient()
db = client['hn-db']
collection = db.items

totalTitle = []
totalId = []

totalVocab_stemmed = []
totalVocab_tokenized = []

totalCorpus = []


# Get posts randomly given the number of posts you need
def getPostsByIdRange(sampleSize, isStory=False):
    randomSet = [int(num) for num in (set(np.random.choice(13000000, sampleSize)))]
    if isStory:
        posts = list(collection.find({"id": {"$in": randomSet}, "type" : "story"}).batch_size(1000))
        print("There are %s items in total" % len(posts))
        for post in posts:
            if ('dead' not in post or post['dead'] == False) and 'title' in post:
                # posts.append(post)
                totalTitle.append(post['title'])
                totalId.append(post['id'])
                corpus = ''.join(constructStory(post))
                allwords_tokenized, allwords_stemmed = tokenize(corpus);
                totalVocab_stemmed.extend(allwords_stemmed)
                totalVocab_tokenized.extend(allwords_tokenized)
                totalCorpus.append(corpus)
        return totalCorpus
    else:
        posts = list(collection.find({"id": {"$in": randomSet}}).batch_size(1000))
        for post in posts:
            if ('dead' not in post or post['dead'] == False) and 'title' in post:
                # posts.append(post)
                totalTitle.append(post['title'])
                totalId.append(post['id'])
                corpus = ''.join(constructStory(post))
                allwords_tokenized, allwords_stemmed = tokenize(corpus)
                totalVocab_stemmed.extend(allwords_stemmed)
                totalVocab_tokenized.extend(allwords_tokenized)
                totalCorpus.append(corpus)
        # posts = [post for post in posts if ('dead' not in post or post['dead'] == False)
        #          and 'title' in post]
        print("There are %s items in total" % len(posts))
        return totalCorpus

def getPostsByAuthor(author):
    for post in collection.find({"by": author}):
        pprint.pprint(post)


def tokenize(text, stem=True):
    # Remove punctuation and numbers
    text = ''.join([ch for ch in text if ch not in string.punctuation and not ch.isdigit()])
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop and len(token) > 1]

    # Extract the stem of each word
    stemmer = EnglishStemmer()
    tokens_stemmed = [stemmer.stem(token) for token in tokens]
    # Remove non-words (based on length)
    # tokens = [stem.lower() for stem in stems if len(stem) < 20 and len(stem) > 1]
    return tokens, tokens_stemmed


# Create a corpus for each given story (including all comments)
# story: array of comments
def createVocabulary(story):
    vocabulary = [tokenize(comment) for comment in story]
    return vocabulary


# Given the parent story id, traverse the tree and construct the
# story object
def constructStory(parent):
    storage = []
    if isinstance(parent, int):
        #     print('Processing id: %s' % parent)
        data = list(collection.find({"id": parent}))
    else:
        data = list(collection.find({"id": parent['id']}))
    if len(data) == 0:
        return []
    if 'title' in data[0]:
        storage.append(data[0]['title'])
    if 'kids' in data[0] and len(data[0]['kids']) > 0:
        children = data[0]['kids']
        for child in children:
            storage.extend(constructStory(child))
    elif 'type' in data[0] and data[0]['type'] == 'comment' and 'text' in data[0]:
        if 'dead' not in data[0] or data[0]['dead'] is False:
            storage.append(data[0]['text'])
    return storage

startTime = time.time()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000, min_df=0.1, stop_words='english', use_idf=True,
                                   ngram_range=(1, 3), tokenizer=tokenize)

print('Vectorization - {}s'.format((time.time() - startTime)))
startTime = time.time()
# Construct a corpus based on 3000 stories
getPostsByIdRange(30, True)

print('getPostsByIdRange - {}s'.format((time.time() - startTime)))
startTime = time.time()
print('Shape of totalVocab_stemmed %s ' % len(totalVocab_stemmed))
print('Shape of totalVocab_tokenized %s ' % len(totalVocab_tokenized))
vocab_frame = pd.DataFrame({'words': totalVocab_tokenized}, index=totalVocab_stemmed)
print('There are %s items in vocab_frame' % vocab_frame.shape[0])

# totalCorpus = ''.join(totalCorpus).split(' ')
# print(totalCorpus)
tfidf_matrix = tfidf_vectorizer.fit_transform(totalCorpus)
print('tfidf_vectorizer.fit_transform - {}s'.format((time.time() - startTime)))
startTime = time.time()
# print(tfidf_matrix)


# %time tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
# print(tfidf_matrix)


# In[45]:
print('-------------- Features -------------')
# Calculate the similarity matrix based on tfidf
terms = tfidf_vectorizer.get_feature_names()
print(terms)
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)
# In[40]:

# Classifiy the posts into clusters
startTime = time.time()
from sklearn.cluster import KMeans

num_clusters = 8
km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
print('kmeans.fit - {}s'.format((time.time() - startTime)))

clusters = km.labels_.tolist()
print('-------------- Clusters -------------')
print(clusters)
print()
# In[41]:

# Access the totalCorpus matrix by using the clusters as the index
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
print(order_centroids)
print('Top posts per cluster: ')

topNWords = 8

for i in range(num_clusters):
    print('---------------- %s ------------------' % i)
    for ind in order_centroids[i, :topNWords]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print()
        print()
    count = 0
    for j in range(0, len(clusters)):
        if clusters[j] == i and count < 10:
            count += 1
            print(totalTitle[j])
        if count == 20:
            break
