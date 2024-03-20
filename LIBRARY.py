import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random    # pseudo-random number generator
import time
import os
import tensorflow


nltk.download('twitter_samples')           # downloads sample twitter dataset.
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')                          # select the set of positive and negative tweets
print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))

print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))


fig = plt.figure(figsize=(5, 5))     # Declare a figure with a custom size

labels = 'Positives', 'Negative'     # labels for the two classes

sizes = [len(all_positive_tweets), len(all_negative_tweets)] # Sizes for each slide

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90) # Declare pie chart, where the slices will be ordered and plotted counter-clockwise


plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


plt.show() # Display the chart
