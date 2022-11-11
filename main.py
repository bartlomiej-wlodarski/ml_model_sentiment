from flask import Flask, render_template, url_for, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))

    return df


def word_stemmer(text):
    stem_text = [PorterStemmer().stem(i) for i in text]
    return stem_text


def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text


def word_pos_tagger(text):
    pos_tagged_text = nltk.pos_tag(text)
    return pos_tagged_text

app = Flask(__name__)

tweet_collection = pd.read_csv('smallProcessed.csv', encoding='ISO-8859-1')

tweet_collection = tweet_collection[["text", "polarity"]]

# Randomize the entire data set
randomized_collection = tweet_collection.sample(frac=1, random_state=3)

# Calculate index for split
training_test_index = round(len(randomized_collection) * 0.9)

# Training/Test split
training_set = randomized_collection[:training_test_index].reset_index(drop=True)
test_set = randomized_collection[training_test_index:].reset_index(drop=True)

# remove noise characters
training_set_clean = clean_text(training_set, "text", "text")

# remove stop words
training_set_clean['text'] = training_set_clean['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))

# tokenization
training_set_clean['text'] = training_set_clean['text'].apply(lambda x: word_tokenize(x))

# stemming
training_set_clean['text'] = training_set_clean['text'].apply(lambda x: word_stemmer(x))

# lemmatization
training_set_clean['text'] = training_set_clean['text'].apply(lambda x: word_lemmatizer(x))

corpus = training_set_clean['text'].sum()

temp_set = set(corpus)

# Revert to a list
vocabulary = list(temp_set)

# Create the dictionary
len_training_set = len(training_set_clean['text'])
word_counts_per_text = {unique_word: [0] * len_training_set for unique_word in vocabulary}

for index, sms in enumerate(training_set_clean['text']):
    for word in sms:
        word_counts_per_text[word][index] += 1

# Convert to dataframe
word_counts = pd.DataFrame(word_counts_per_text)

# Concatenate with the original training set
training_set_final = pd.concat([training_set_clean, word_counts], axis=1)

# Filter the dataframes
"""nie rozpoznaje różnicy między 2 a 4 i daje takie tam takie same wyrazy
jak zmienisz cyfry na stringi 0 -> '0' to nie wyjebie errora później, ale wtedy nie widzi w ogóle różnicy"""
negative = training_set_final[training_set_final['polarity'] == 0].copy()
neutral = training_set_final[training_set_final['polarity'] == 2].copy()
positive = training_set_final[training_set_final['polarity'] == 4].copy()

# Calculate
"""print("negative", negative)
print("neutral;", neutral)
print("positive", positive)
print("negative shape", negative.shape)
print("negative shape[1]", negative.shape[1])
print("set shape", training_set_final.shape)
print("set shape[0]", training_set_final.shape[0])"""
p_negative = negative.shape[1] / training_set_final.shape[0]
p_neutral = neutral.shape[1] / training_set_final.shape[0]
p_positive = positive.shape[1] / training_set_final.shape[0]
""" te zmienne są takie same a chyba nie powinny"""

# Calculate vocabulary
negative_words_per_message = negative['text'].apply(len)
n_negative = negative_words_per_message.sum()

neutral_words_per_message = neutral['text'].apply(len)
n_neutral = neutral_words_per_message.sum()

positive_words_per_message = positive['text'].apply(len)
n_positive = positive_words_per_message.sum()

n_vocabulary = len(vocabulary)

alpha = 1

# Create three dictionaries that match each unique word with the respective probability value.
parameters_negative = {unique_word: 0 for unique_word in vocabulary}
parameters_neutral = {unique_word: 0 for unique_word in vocabulary}
parameters_positive = {unique_word: 0 for unique_word in vocabulary}

# Iterate over the vocabulary and for each word, calculate
for unique_word in vocabulary:
    p_unique_word_negative = (negative[unique_word].sum() + alpha) / (n_negative + alpha * n_vocabulary)
    p_unique_word_neutral = (neutral[unique_word].sum() + alpha) / (n_neutral + alpha * n_vocabulary)
    p_unique_word_positive = (positive[unique_word].sum() + alpha) / (n_positive + alpha * n_vocabulary)

    # Update the calculated probabilities to the dictionaries
    parameters_negative[unique_word] = p_unique_word_negative
    parameters_neutral[unique_word] = p_unique_word_neutral
    parameters_positive[unique_word] = p_unique_word_positive


def text_classify(message):
    message = message.lower()
    message = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", message)
    # remove numbers
    message = re.sub(r"\d+", "", message)

    # remove stop words
    message = ' '.join([word for word in message.split() if word not in (stopwords.words('english'))])

    # tokenization
    message = word_tokenize(message)

    # stemming
    message = word_stemmer(message)

    # lemmatization
    message = word_lemmatizer(message)

    p_negative_message = p_negative
    p_neutral_message = p_neutral
    p_positive_message = p_positive

    for word in message:
        if word in parameters_negative:
            p_negative_message *= parameters_negative[word]

        if word in parameters_neutral:
            p_neutral_message *= parameters_neutral[word]

        if word in parameters_positive:
            p_positive_message *= parameters_positive[word]

    print(p_negative_message)
    print(p_neutral_message)
    print(p_positive_message)

    """ tutaj trzy zmienne są takie same, więc nie ma jak podjąć decyzji """


print(text_classify("I'm so pissed off I wanna kill things, murder fuck"))
print(text_classify("That makes me happy, I'm glad everything is fine"))

print(p_negative)
print(p_neutral)
print(p_positive)

