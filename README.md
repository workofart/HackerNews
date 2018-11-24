# HackerNews - Data mining and analysis

#### Tagging posts based on Natural Language Processing on the post title and comments.

[Front-end Page](https://github.com/workofart/HN-data-analysis) for interacting with the data.

---

###### Current approach:
- LDA (Latent Dirichlet Allocation)
- Custom tokenizer
- Unsupervised learning
- 24 topics
- 70000 stories input
- 242591 unique tokens
- Removed top 30 most frequent words
- Trained on Multicore - 3 workers
- 100 passes

Topics (Personally labeled the topics by the top 20 keywords in each topic):
- Social
- Startups
- Science
- Business/Finance
- Software Development
- Technology
- Society/Economics
- Software
- Entertainment
- Web Infrastructure/Technologies
- Innovation
- Finance/Stock Market
- Education
- Programming
- Security
- Websites
- Resource
- Health/Body
- Database
- Product/Startup
- Legal
- Politics
- Web
- Job

---

**Assumptions:**
- MongoDB set up already (Crawled HackerNews using their API and stored 12862100 stories + comments into the DB)
- Story and comments are stored in database called 'hn-db', in 'items' collection.
- Each record in the DB is either a story or a comment


![DB Example](https://raw.githubusercontent.com/workofart/HackerNews/master/db.png)


**dataset_utils.py**
- provides the functions for sampling from the DB
- constructing a tree structure instead of individual comments being in separate dictionaries
    
    
    E.g.
    
        {
            _id:587d843989f43582597791d7,
            type:"story",
            id:1599998,
            title:"Yet Another Article on Zippers, in Erlang",
            kids: [
                    {
                      "_id" : ObjectId("587d85ab89f43582590a53f5"),
                      "type" : "comment",
                      "id" : 1602275,
                      "text" : "I stopped reading where it says that lists are \"O(n) (although they can average n/2.)\"",
                      "parent" : 1599998,
                      "time" : 1281733168,
                      "by" : "nemoniac"
                    }
                ],
            time:1281670730,
            text:"",
            score:11,
            url:"http://ferd.ca/yet-another-article-on-zippers.html",
            by:"mononcqc",
            descendants:1
        }


**analysis_utils.py**
- Term Frequency Inverse Document Frequency Implementation
- Custom tokenizer
- Word2Vec Train/Test
- kMeans Test
- LDA Train/Test
- Add tags to all stories in another table/collection in the DB
