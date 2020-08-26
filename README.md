# Twitter Sentiment Analysis

This is an opinion mining program that predicts the sentiment for a search term by retrieving tweets that contain that search term and analysing them. In order to perform some rudimentary error analysis, another opinion miner was built on top using TextBlob to capture the 'true' sentiment of each tweet and therefore the search term overall. The model being used here is a Naive Bayes Classifier (for both the predicted and the 'true' sentiment). The word 'true' is in inverted commas because it is not known whether TextBlob definitely predicts the correct sentiment, but it is more accurate than the prediction and so has been taken to be used as a benchmark.

In order to run this script one can simply import the file (making sure everything is in the same directory). The training tweets have been uploaded to save time, but one can build their own training set using the TrainingTweetsBuilder script if they wish (this will take a long time). The corpus csv file is necessary to run the training tweets builder.
The API authentication is specific, and new credentials will be required to authenticate the program to twitter. Information on how to do this is well outlined on the article in the URL below.

This script was created by following an article on towardsdatascience written by Anas Al-Masri (the URL is at the bottom). It's self-contained, but it’s very dated and so some changes needed to be made to the code. Other instructions were altered to make the code cleaner and run more efficiently.

In the article, there is no error analysis performed. My own error analysis is completely ad-hoc and self-inspired.


https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed
