import unittest
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from main import clean_text, word_stemmer, word_lemmatizer, word_pos_tagger, text_classify

#co jest naszym df?
class TestCleanTextMethod(unittest.TestCase):
    def test_clean_text(self):
        result = clean_text(df, "Play!123", "play")
        self.assertEqual(result, df + "play" + "play")

class TestWordStemmerMethod(unittest.TestCase):
    def test_word_stemmer(self):
        #result = word_stemmer("playing")
        text = 'playing'
        word_stemmer(text)
        stem_text = [PorterStemmer().stem(i) for i in text]
        self.assertEqual(stem_text, 'play')

class TestWordLemmatizerMethod(unittest.TestCase):
    def test_word_lemmatizer(self):
        text = 'better'
        word_lemmatizer(text)
        lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
        self.assertEqual(lem_text, 'good')


#tego nie wiem jak przetestować, jaki output powinna nam ta metoda dawać
#class TestTextClassifyMethod(unittest.TestCase):
#    def test_text_classify(self):


if __name__ == '__main__':
    unittest.main()
