from pymongo import MongoClient
import pprint, json

##### DB Settings #####
client = MongoClient()
db = client['hn-db']
collection = db.items
target_table = db.tree
visitedItems = []


# Given the parent story id, traverse the tree and construct the
# story object
def constructStory(parent):
    storage = []
    if isinstance(parent, int):
        #     print('Processing id: %s' % parent)
        data = list(collection.find({"id": parent}))
        visitedItems.append(parent)
    else:
        data = list(collection.find({"id": parent['id']}))
    if len(data) == 0:
        return []
    if 'title' in data[0]:
        storage.append(data[0])
    if 'kids' in data[0] and len(data[0]['kids']) > 0:
        children = data[0]['kids']
        for child in children:
            storage.extend(constructStory(child))

    # Comment Case
    elif 'type' in data[0] and data[0]['type'] == 'comment' and 'text' in data[0]:
        if 'dead' not in data[0] or data[0]['dead'] is False:
            storage.append(data[0])
    return storage

# for i in range(0, 100):
#     if i not in visitedItems:
#         print(constructStory(i))
#         print('------------------------------')



dataMap = {}
tree = []
UPPER_LIMIT = 20000
INCREMENT = 1000
# for i in range(0,2):
for i in range(0, int(UPPER_LIMIT/INCREMENT)):
    lower = i * INCREMENT
    upper = (i + 1) * INCREMENT + 1
    print( str(lower) + ' < ' + ' i ' + ' < ' + str(upper) )
    itemList = list(collection.find({"id": { "$gt" : lower, "$lt" : upper}}, {"_id" : 0}).batch_size(5000))
    for item in itemList:
        if ('kids' in item):
            item['kids'] = [kids for kids in item['kids'] if kids < UPPER_LIMIT]
        dataMap[item['id']] = item
        if ('parent' in item and item['parent'] in dataMap):
            parent = dataMap[item['parent']]
            parent['kids'] = []
            if ('kids' in parent):
                # Create kids array if it doesn't exist
                # if (parent['kids'] is None):
                #     parent['kids'] = []
                if (item['id'] in parent['kids']):
                    parent['kids'].remove(item['id'])
                if item['id'] < UPPER_LIMIT:
                    # print('Appending {}'.format(item['id']))
                    parent['kids'].append(item)
        else:
            # result = target_table.insert_one(item)
            # print('Inserted ' + str(result.inserted_id))
            tree.append(item)
#
# tmp = 12860000
# i = 12862101
# print(str(tmp) + ' < ' + ' i ' + ' < ' + str(i))
# itemList = list(collection.find({"id": { "$gt" : tmp, "$lt" : i}}, {"_id" : 0}).batch_size(5000))
# for item in itemList:
#     dataMap[item['id']] = item
#     if ('parent' in item and item['parent'] in dataMap):
#         parent = dataMap[item['parent']]
#         if ('kids' in parent):
#             # Create kids array if it doesn't exist
#             if (parent['kids'] is None):
#                 parent['kids'] = []
#             if (item['id'] in parent['kids']):
#                 parent['kids'].remove(item['id'])
#             parent['kids'].append(item)
#     else:
#         # result = target_table.insert_one(item)
#         # print('Inserted ' + str(result.inserted_id))
#         tree.append(item)
# print('Finished building the data map')


# print(dataMap)
# print(itemList)
#
print('Finished creating the tree')

with open('HN-tree.json', 'w') as file:
    json.dump(tree, file, indent=4)