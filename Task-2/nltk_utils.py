import nltk
#nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sen):
    return nltk.word_tokenize(sen)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    '''
    example that how bag_of words work:
    
    
    tokenized_sentence = ["hello","how","are","you"]
    all_words = ["hi","hello","I","you","bye","thanks","cool"]
    bag=        [0,     1,     0,   1,    0,     0,      0   ]
    
    
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words),dtype=np.float32)
    
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

'''
sent = ["hello","how","are","you"]
all_words = ["hi","hello","I","you","bye","thanks","cool"]

print(bag_of_words(sent,all_words))

'''


