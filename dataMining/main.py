import _pickle, json, requests, html, datetime, os, threading, time, ijson, ujson

############# Parameters #############
fileName = 'WhosHiringJan2017'
fileNameTest = 'Test'
item = 13301832
itemTest = 13142647

today = datetime.datetime.today()
today_date = str(today.year).zfill(4) + str(today.month).zfill(2) + str(today.day).zfill(2)


## Todo:
# Ranking of stories that are worth investigating
# - Decendents (number of comments in total for this story)
# - Score (complex derived number)

# Create a front-end web-page that can easily navgiate through the stories/asks
# - Sort functions (sort comments within a story)
# - Summarizing functions
# - Categorizing functions
# - Sentiment functions


# Create an entry-point to understanding user behavior towards a certain topic/product
# - word frequency that relate to the dimensions of the product
    # E.g. For cars, handling would be one dimension, and the frequency of 'handling'
    # Can be used to describe the "importance" of that dimension




# 13139638 - Ask HN: What problem in your industry is a potential startup?


# TODO: Max retries exceeded with url
def getAllStories(fromId, toId):
    lastStoryId = requests.get('https://hacker-news.firebaseio.com/v0/maxitem.json?print=pretty').json()
    currentStoryId = toId
    while (currentStoryId > fromId):
        currentStoryId = currentStoryId - 1
        url = 'https://hacker-news.firebaseio.com/v0/item/' + str(currentStoryId) + '.json?print=pretty'
        allStoriesDir = 'data/allStories'
        try:
            item = requests.get(url).json()
            # if os.path.isdir(allStoriesDir) == False:
            #     os.mkdir(allStoriesDir)
            writeToFile(json.dumps(item, indent=4))
            # with open(allStoriesDir + '/' + str(currentStoryId) + '.json', 'w') as f:
                # json.dump(item, f, indent=4)
        except requests.exceptions.ConnectionError:
            print('Connection Refused, try in 60 seconds')
            time.sleep(60)



    print('done fromId: ' + currentStoryId)

def getTopStories():
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty'
    topStoryDir = 'data/top_stories_' + today_date
    itemSet = requests.get(url).json()
    i = 0
    if os.path.isdir(topStoryDir) == False:
        os.mkdir(topStoryDir)
    for item in itemSet:
        i += 1
        print('Processing [' + str(i) + '] top story ' + str(item))
        with open(topStoryDir + '/' + str(item) + '.json', 'w') as f:
            json.dump(crawlKids(item, 0), f, indent=4)


def getTopAsks():
    url = 'https://hacker-news.firebaseio.com/v0/askstories.json?print=pretty'
    topAskDir = 'data/top_asks_' + today_date
    itemSet = requests.get(url).json()
    i = 0
    if os.path.isdir(topAskDir) == False:
        os.mkdir(topAskDir)
    for item in itemSet:
        i += 1
        print('Processing [' + str(i) + '] top asks: ' + str(item))
        with open(topAskDir + '/' + str(item) + '.json', 'w') as f:
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

def isDeleted(jsonObj):
    if 'deleted' in jsonObj.json():
        return jsonObj.json()['deleted']
    else:
        return None

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

            # Make sure the kid has comment element and is not deleted
            if res is None or res.json() is None:
                print('kid: ' + str(kid) + ' is None')
            elif 'type' in res.json() and getType(res) == 'comment' and isDeleted(res) is None:
                    kidStorage.append(crawlKids(kid, depth+1))

        storage['kids'] = kidStorage
        return storage

    # print('Parent type: ' + getType(getResObj(getParent(res))))
    # print(kids)


def getOneStory(item):
    with open(fileName + '.json', 'w') as f:
        json.dump(crawlKids(item, 0), f, indent=4)

class myThread(threading.Thread):
    def __init__(self, threadID, name, counter, fromId, toId):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.fromId = fromId
        self.toId = toId

    def run(self):
        print
        "Starting " + self.name
        # print_time(self.name, self.counter, 5)
        getAllStories(self.fromId, self.toId)
        print
        "Exiting " + self.name

lock = threading.Lock()

def writeToFile(item):
    lock.acquire()

    with open('combined_part2.json', "a") as outfile:
        outfile.write(',')
        outfile.write(item)
    lock.release()

# Create new threads
max_threads = 100

def startPulling(max_threads):
    for x in range(0, max_threads):
        thread = myThread(x, "Thread-" + str(x), x, x * 30000 + 10000000, (x + 1) * 30000 + 10000000)
        thread.start()

startPulling(max_threads)
# with open('combined.json' , 'r') as inFile:
#     combined = ijson.items(inFile, 'type');
#     chunkSize = 1000000
#     i = 0
#     for o in combined:
#         i = i + 1
#     for i in range(0, 10000000, chunkSize):
#         print('Starting chunk: ' + str(i))
#         with open('part_' + str(i) + '.json' , 'w') as outfile:
#             ujson.dump(combined[i:i+chunkSize], outfile)

# thread1 = myThread(1, "Thread-1", 1, 0, 1000000)
# thread2 = myThread(2, "Thread-2", 2, 1000001, 2000000)
# thread3 = myThread(3, "Thread-3", 3, 2000001, 3000000)
# thread4 = myThread(4, "Thread-4", 4, 3000001, 4000000)
# thread5 = myThread(5, "Thread-5", 5, 4000001, 5000000)
# thread6 = myThread(6, "Thread-6", 6, 5000001, 6000000)
# thread7 = myThread(7, "Thread-7", 7, 6000001, 7000000)
# thread8 = myThread(8, "Thread-8", 8, 7000001, 8000000)
# thread9 = myThread(9, "Thread-9", 9, 8000001, 9000000)
# thread10 = myThread(10, "Thread-10", 10, 9000001, 10000000)

# Start new Threads
# thread1.start()
# thread2.start()
# thread3.start()
# thread4.start()
# thread5.start()
# thread6.start()
# thread7.start()
# thread8.start()
# thread9.start()
# thread10.start()


# getTopAsks()
# getTopStories()
# _thread.start_new_thread(getAllStories(0, 1000))
# _thread.start_new_thread(getAllStories(1001, 2000))
# _thread.start_new_thread(getAllStories(2001, 3000))
# getAllStories()

# getOneStory(item)

