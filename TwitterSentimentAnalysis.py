import twitter
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
from textblob import TextBlob 
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')

#Authenticating Twitter App (credentials will need to be changed other users)
twitter_api = twitter.Api(consumer_key ='8i68MvUnFgp3Mqh5jRV34jGh3',
                        consumer_secret= 'jNPsW7ZKZkKoVLAvFjwgTPHvlw4Hto8jbpnCDcbnnSUm7O8RJX',
                        access_token_key='2378518193-5lkOIoa6OJpm3X0AehYz9EAjws6NjSsjHZS93MM',
                        access_token_secret = 'DN6j3LxuKWtkKJkGpU16pUr2wqSFYoUs6FwED1SXFjpAw')
#-------------------------------------------------------------------------------------------
#Building the test data set
def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count = 100, lang = "en")
        
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        return None
    pass
#-------------------------------------------------------------------------------------------
#Building the training data set
"""This is done in a seperate file"""
import TrainingTweetsBuilder
#----------------------------------------------------------------------------------------------
    
#Access Training tweets
def AccessCSV(train_data):
    trainDataSet = [] 
    with open(train_data,'r', encoding = "utf-8") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
        for row in lineReader:
                trainDataSet.append(row)
    return trainDataSet
  
#--------------------------------------------------------------------------------------------------
#Create and implement PreProcessor
class PreProcessTweets:
    
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            if any(isinstance(tweet,dict) for tweet in list_of_tweets):
                try:
                    processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
                except Exception as e:
                    print(e)
            else:
                try:
                    processedTweets.append((self._processTweet(tweet[1]),tweet[2]))
                except Exception as e:
                    print(e)
                
        return processedTweets
    
    def _processTweet(self, tweet):
       tweet = tweet.lower() # convert text to lower-case
       tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
       tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
       tweet =  re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
       tweet =  word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
       return [word for word in tweet if word not in self._stopwords]
#---------------------------------------------------------------------------------------------------       

#Prepping the model
def buildVocabulary(data_to_train):
    all_words = []
    
    for (words, sentiment) in data_to_train:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    return word_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features 

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


#The entire algorithm

#Build test set
    
search_term = input("Search on Twitter:")
testDataSet = buildTestSet(search_term)
print(testDataSet[:5])

print ("  ")

#Access saved training set
train_data =  "tweetDataFile.csv"     
trainDataSet = list(filter(None, AccessCSV(train_data)))

#Clean the training and test data, prepare features (words in the tweets) and target (sentiment)
tweetProcessor = PreProcessTweets()
data_to_train = tweetProcessor.processTweets(trainDataSet)
data_to_test = tweetProcessor.processTweets(testDataSet)
word_features = buildVocabulary(data_to_train)
trainingFeatures = nltk.classify.apply_features(extract_features, data_to_train)
         
#Train the learning model (in this case, a Naive Bayes Classifier) and apply testing tweets to test tweets 
NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in data_to_test]

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#Error Analysis

#Get the correct sentiment of the test tweets using TextBlob and a copy of the test data
#Save the label count
X = 0
Y = 0 
Z = 0 
testDataSet_copy = testDataSet.copy()
for tweet in testDataSet_copy:
    if  TextBlob(tweet['text']).sentiment.polarity > 0:
        tweet['label'] = "positive"
        X+= 1
    elif TextBlob(tweet['text']).sentiment.polarity < 0:
        tweet['label'] = "negative"
        Y += 1
    else:
        tweet['label'] = "neutral"
        Z += 1
     
#Calculate the correct overall sentiment
sentiment = []
for tweet in testDataSet_copy:
    sentiment.append(TextBlob(tweet['text']).sentiment.polarity)
real_sentiment = np.mean(sentiment)
    
#Measure the accuracy of predections
matches = 0
total = len(testDataSet)
for i, label in enumerate(NBResultLabels):
    if label == testDataSet_copy[i]['label']:
        matches += 1
accuracy = 100*(matches/total)

#Print results

#get the predicted sentiment
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Predicted: Overall Positive Sentiment")
    print(f"Positive Sentiment Percentage = {100*NBResultLabels.count('positive')/len(NBResultLabels)}%")
else: 
    print("Predicted: Overall Negative Sentiment")
    print(f"Negative Sentiment Percentage = {100*NBResultLabels.count('negative')/len(NBResultLabels)}%")

print ("  ")

#get the real sentiment
if real_sentiment > 0:
    print ('Actaul: Overall Positive sentiment')
    print(f"Positive Sentiment Percentage = {100*X/total}%")
    
elif real_sentiment < 0:
    print ("Actual: Overall Negative Sentiment")
    print(f"Negative Sentiment Percentage = {100*Y/total}%")

else:
    print (f"Actual: Overal Neutral Sentment with a percentage of {100*Z/total}%'")
    
print("  ")
print(f"The sentiment of {matches} out of {total} tweets was predicted correctly")
print(f'Accuracy of prediction = {round(accuracy,2)}%')
    

        