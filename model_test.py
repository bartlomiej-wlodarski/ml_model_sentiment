import unittest
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from main import clean_text, word_stemmer, word_lemmatizer, text_classify, test_set
import pandas as pd


class TestCleanTextMethod(unittest.TestCase):
    def test_clean_text(self):
        data = {'text field': ['Play!123']}
        df = pd.DataFrame(data)
        data = {'text field': ['play']}
        exdf = pd.DataFrame(data)
        result = clean_text(df, "text field", "text field")
        self.assertEqual(exdf.at[0, "text field"], result.at[0, "text field"])

class TestWordStemmerMethod(unittest.TestCase):
    def test_word_stemmer(self):
        #result = word_stemmer("playing")
        text = 'playing'
        text = word_tokenize(text)
        text = word_stemmer(text)
        self.assertEqual(word_tokenize('play'), text)

class TestWordLemmatizerMethod(unittest.TestCase):
    def test_word_lemmatizer(self):
        text = 'rocks'
        text = word_tokenize(text)
        text = word_lemmatizer(text)
        self.assertEqual(text, word_tokenize('rock'))


class TestTextClassifyMethod(unittest.TestCase):
    def test_text_classify(self):
        self.assertEqual(text_classify("I'm so pissed off I wanna kill things, murder hatred"), -1)
        self.assertEqual(text_classify("That makes me happy, I'm glad everything is fine"), 1)

class TestTextClassifyAccuracy(unittest.TestCase):
    def test_classify_accuracy(self):
        matches = 0
        for index, row in test_set.iterrows():
            if text_classify(row['text field']) == -1 and row['polarity field'] == 0:
                matches = matches + 1
            elif text_classify(row['text field']) == 1 and row['polarity field'] == 4:
                matches = matches + 1
        print("accuracy ", matches / test_set.shape[0] * 100, "%")
        self.assertGreater(matches / test_set.shape[0], 0.8)

if __name__ == '__main__':
    unittest.main()
