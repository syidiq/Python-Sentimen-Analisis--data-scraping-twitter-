import string
import collections

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import ast
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

data = pd.read_csv("DT1.csv", encoding='latin-1', sep=",")


text_list = data['Text']
vectorizer = TfidfVectorizer(max_df=0.5,min_df=0.1,lowercase=True)
 
tfidf_model = vectorizer.fit_transform(text_list)
km_model = KMeans(n_clusters=2)
km_model.fit(tfidf_model)
clustering = collections.defaultdict(list)
 
for idx, label in enumerate(km_model.labels_):
    clustering[label].append(idx)
data['topik'] = None
for i in clustering[0]:
    data['topik'][i]=1
for j in clustering[1]:
    data['topik'][i]=2
    
data.to_csv (r'C:\Users\MSA\Desktop\Data\data_cls.csv', index = False, header=True)
