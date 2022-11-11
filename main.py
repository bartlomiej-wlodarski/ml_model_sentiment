from flask import Flask,render_template,url_for,request
import pandas as pd
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

tweet_collection = pd.read_csv('trainProcessed.csv', encoding='ISO-8859-1')

tweet_collection = tweet_collection[["text", "polarity"]]

# Randomize the entire data set
randomized_collection = tweet_collection.sample(frac=1, random_state=3)

# Calculate index for split
training_test_index = round(len(randomized_collection) * 0.9)

# Training/Test split
training_set = randomized_collection[:training_test_index].reset_index(drop=True)
test_set = randomized_collection[training_test_index:].reset_index(drop=True)

# remove noise characters
remove_characters = ["$", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                     "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+",
                     "[", "]", "{", "}", ";", ":", "?", ",", ".", "/", "|", "'",
                     "\"", "\n", "\t", "\r", "\b", "\f", "\\", "\v", "_"]

for character in remove_characters:
    training_set["text"] = training_set["text"].replace(character, "")

print(training_set.head())