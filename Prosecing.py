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

import warnings
warnings.filterwarnings("ignore")

np.random.seed(300)


data = pd.read_csv("coba.csv", encoding='latin-1', sep=";")


print(data.head(10))

n=data['Text'].count()
symbols = "!\"#$%&()*+-.,/:;'<=>?@[\]^_`{|}~\n"
for i in range(0,n):
    data['Text'][i] = data['Text'][i].lower()
    data['Text'][i] = re.sub(r"\d+", "", data['Text'][i])
    data['Text'][i] = data['Text'][i].translate(str.maketrans("","",symbols))
    data['Text'][i] = data['Text'][i].strip()
    data['Text'][i] = word_tokenize(data['Text'][i])
    listStopword =  set(stopwords.words('indonesian'))
    listStopword.update(['rt','pak','bapak''yg','ga','ya','subtanyarl','biar','aja','orang','gak','sbyfess','wkwk','udah','iya','a','lupa','kalo','nya','banget','bgt','sih','tau','kak','i','rynryd','dipelajari','httpstcouqsntzdjpp','suka','pagi','nih','gue','the','mas','gitu','you','ku','bikin','si','teman','pake','semoga','to','anak','tp','pas','bilang','for','and'])
    dd = data['Text'][i]
    data['Text'][i] = []
    for t in dd:
        if t not in listStopword:
            data['Text'][i].append(t)
    print('looping ke-',i)
print(data.head(10))
data.to_csv (r'C:\Users\MSA\Desktop\Data\export_dataframe.csv', index = False, header=True)
