import re
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import ast

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("newdata.csv", encoding='latin-1', sep=",")

def convert_text_list(texts):
    texts = ast.literal_eval(texts)
    return [text for text in texts]

data['Text'] = data['Text'].apply(convert_text_list)

def calc_TF(document):
    # Counts the number of times the word appears in review
    TF_dict = {}
    for term in document:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1
    # Computes tf for each word
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(document)
    return TF_dict
#d = data['Text'].apply(calc_TF)
data["TF_dict"] = data['Text'].apply(calc_TF)


def calc_DF(tfDict):
    count_DF = {}
    # Run through each document's tf dictionary and increment countDict's (term, doc) pair
    for document in tfDict:
        for term in document:
            if term in count_DF:
                count_DF[term] += 1
            else:
                count_DF[term] = 1
    return count_DF
DF = calc_DF(data["TF_dict"])
n_document = len(data)
def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict
IDF = calc_IDF(n_document, DF)


def calc_TF_IDF(TF):
    TF_IDF_Dict = {}
    #For each word in the review, we multiply its tf and its idf.
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict
#dd = d.apply(calc_TF_IDF)
data["TF-IDF_dict"] = data["TF_dict"].apply(calc_TF_IDF)


# sort descending by value for DF dictionary 
sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)

# Create a list of unique words from sorted dictionay `sorted_DF`
unique_term = [item[0] for item in sorted_DF]

def calc_TF_IDF_Vec(__TF_IDF_Dict):
    TF_IDF_vector = [0.0] * len(unique_term)

    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector

#ddv = dd.apply(calc_TF_IDF_Vec)
data["TF_IDF_Vec"] = data["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

# Convert Series to List
TF_IDF_Vec_List = np.array(data["TF_IDF_Vec"].to_list())

# Sum element vector in axis=0 
sums = TF_IDF_Vec_List.sum(axis=0)

dt = []

for col, term in enumerate(unique_term):
    dt.append((term, sums[col]))
    
ranking = pd.DataFrame(dt, columns=['term', 'rank'])
RS = ranking.sort_values('rank', ascending=False)

