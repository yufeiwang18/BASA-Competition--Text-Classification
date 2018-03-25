import pandas as pd
import numpy as np
import re,os,glob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stemmer.porter import PorterStemmer

#stemming
from nltk.stem.porter import PorterStemmer
def remove_stem(list):
    new_list=[]
    for row in list:
        new_row=[]
        for i in row:
            new_row.append(PorterStemmer().stem(i))
        new_list.append(new_row)
    return new_list


# remove stopwords
stop_words = set(stopwords.words('english'))
def remove_sw(string):
    temp=tokenizer.tokenize(string)
    return [w for w in temp if not w in stop_words]

# remove all PUNCTUATION
tokenizer = RegexpTokenizer(r'\w+') #PUNCTUATION = (';', ':', ',', '.', '!', '?')
def tokenize_no_punc(string):
    return ' '.join(tokenizer.tokenize(string))

# data cleaning
def data_clean(df):
    df['description'] = df['description'].apply(tokenize_no_punc)
    df['description'] = df['description'].apply(remove_sw)
    df['description'] = remove_stem(df['description'].tolist())
    return df


# data clean on training dataset
file = os.getcwd()+'/data/training.csv'
data = pd.read_csv(file)
data_tokenized_no_punc = data
data_tokenized_no_punc = data_clean(data_tokenized_no_punc)
data_tokenized_no_punc.to_csv('training_tokenized_no_punc.csv')
print("data clean on training dataset done")

# data clean on testing dataset
file = os.getcwd()+'/data/testing.csv'
data = pd.read_csv(file)
data_tokenized_no_punc = data
data_tokenized_no_punc = data_clean(data_tokenized_no_punc)
data_tokenized_no_punc.to_csv('training_tokenized_no_punc.csv')
print("data clean on testing dataset done")
