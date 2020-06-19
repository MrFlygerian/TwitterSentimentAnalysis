import twitter


#Authenticating Twitter App
twitter_api = twitter.Api(consumer_key ='8i68MvUnFgp3Mqh5jRV34jGh3',
                        consumer_secret= 'jNPsW7ZKZkKoVLAvFjwgTPHvlw4Hto8jbpnCDcbnnSUm7O8RJX',
                        access_token_key='2378518193-5lkOIoa6OJpm3X0AehYz9EAjws6NjSsjHZS93MM',
                        access_token_secret = 'DN6j3LxuKWtkKJkGpU16pUr2wqSFYoUs6FwED1SXFjpAw')

def buildTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time
    
    corpus = []
    
    with open(corpusFile,'r', encoding = "utf-8") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2], "label":row[1], "topic":row[0]})
            
    rate_limit = 180
    sleep_time = 900/rate_limit
    
    trainingDataSet = []
    
    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched: " + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            time.sleep(sleep_time) 
        except: 
            continue
    # now we write the training tweets to the empty CSV file
    with open(tweetDataFile,'w', encoding =  "utf-8") as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet

#Build training set, save set for reuse
corpusFile = "corpus.csv"
tweetDataFile = "tweetDataFile.csv"
trainingData = buildTrainingSet(corpusFile, tweetDataFile)