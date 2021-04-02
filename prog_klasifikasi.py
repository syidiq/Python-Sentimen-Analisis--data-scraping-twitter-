import re
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
import ast
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data_cls.csv", encoding='latin-1', sep=",")
def convert_text_list(texts):
    texts = ast.literal_eval(texts)
    return [text for text in texts]

data['Text'] = data['Text'].apply(convert_text_list)
n=data['Text'].count()
for i in range(0,n):
    dd = data['Text'][i]
    data['Text'][i] = []
    data['Text'][i] = ' '.join(dd)
    print('looping ke-',i)
    data['topik'][i] = int(data['topik'][i])
print(data.head(10))

data.to_csv (r'C:\Users\MSA\Desktop\Data\frame.csv', index = False, header=True)
# Word Vectorization

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data['Text'],data['topik'],test_size=0.3)
Count_vect = CountVectorizer(max_features=50000)
Count_vect.fit(data['Text'])
Train_X_Count = Count_vect.transform(Train_X)
Test_X_Count = Count_vect.transform(Test_X)
print(Count_vect.vocabulary_)

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Count,Train_Y)
predictions_NB = Naive.predict(Test_X_Count)
print("Naive Bayes Accuracy Score ->",accuracy_score(predictions_NB, Test_Y)*100)

knn = KNN(n_neighbors=2)
clf = knn.fit(Train_X_Count,Train_Y)
predicted = clf.predict(Test_X_Count)
print("KNN Accuracy Score ->",accuracy_score(predicted, Test_Y)*100)
