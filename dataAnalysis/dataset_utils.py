import numpy as np
import json
from tqdm import tqdm
from pymongo import MongoClient


ROOTDIR = 'D:\Github\HackerNews'


##### DB Settings #####
client = MongoClient()
db = client['hn-db']
collection = db.items

storedStories = []

# Given the parent story id, traverse the tree and construct the
# story object
def constructStory(id, parent):
    if isinstance(id, int):
        data = collection.find_one({"id": id, "deleted" : { "$exists" : False }, "dead" : { "$exists" : False }}, {"_id" : 0})
        if data is None:
            parent['kids'].remove(id)
            return
    else:
        data = collection.find_one({"id": id['id'], "deleted" : { "$exists" : False }, "dead" : { "$exists" : False }}, {"_id" : 0})
    # print('Appending kid %s' % parent)

    if 'type' in data and data['type'] == 'comment' and 'text' in data:
        if 'kids' in parent and data['id'] in parent['kids']:
            parent['kids'].remove(data['id'])
            parent['kids'].append(data)
        # else:
            # print('%s not in %s' % (data['id'], parent['id']))



    # If there are kids, loop through kids
    if 'kids' in data and len(data['kids']) > 0:
        children = data['kids']
        for i in range(len(children)):
            child = children[0]
            if isinstance(child, int):
                # print('Parent %s | Child %s' % (id, child))
                constructStory(child, data)
            # else:
                # print('Child [%s} is processed already' % child['id'])
            # storage.extend(constructStory(child))
    return data

def export(data, file):
    with open(file, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4, ensure_ascii=True)

tree = []
dataMap = {}
def constructTree(id, UPPER_LIMIT):
    items = list(collection.find_one({"id" : id, "deleted" : { "$exists" : False }, "dead" : { "$exists" : False }}, {"_id" : 0}))
    for item in items:
        if ('kids' in item):
            item['kids'] = [kids for kids in item['kids'] if kids < UPPER_LIMIT]
        for kid in item['kids']:
            # print('Querying for [%d] ' % kid)
            childItem = list(collection.find_one({"id" : id, "deleted" : { "$exists" : False }, "dead" : { "$exists" : False }}, {"_id" : 0}))[0]
            dataMap[childItem['id']] = childItem
        # dataMap[item['id']] = item
        if ('parent' in item and item['parent'] in dataMap):
            # print('Current dataMap: {}'.format(dataMap))
            parent = dataMap[item['parent']]
            parent['kids'] = []
            if ('kids' in parent):
                # Create kids array if it doesn't exist
                # if (parent['kids'] is None):
                #     parent['kids'] = []
                if (item['id'] in parent['kids']):
                    # print('Removing kid [%d]' % item['id'])
                    parent['kids'].remove(item['id'])
                if item['id'] < UPPER_LIMIT:
                    # print('Appending {}'.format(item['id']))
                    parent['kids'].append(item)
        else:
            # result = target_table.insert_one(item)
            # print('Inserted ' + str(result.inserted_id))
            tree.append(item)

def pickSamples(n):
    randomIds = np.random.choice(10000000, n, replace=False)
    for id in tqdm(randomIds):
        data = collection.find_one({'id': int(id), "deleted" : { "$exists" : False }, "dead" : { "$exists" : False }})
        if (data is not None
            and data['type'] == 'story'
            and 'title' in data):
            # print('Processing id: %d' % int(id))
            storedStories.append(constructStory(int(id), {}))

    export(storedStories, 'training_mini_' + str(n) + '.json')

# pickSamples(500000)
# print(collection.find_one({'id' : 7974249}))
