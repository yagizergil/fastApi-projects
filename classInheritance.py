#create a model with 4 classification algorithm.
import re
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from fastapi import FastAPI
from pydantic import BaseModel
import string
string.punctuation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

df=pd.read_csv('all-data7.csv')
df.columns = ["noname","sira","sira2","TEXT_IN_ENGLISH","label"]
df.drop('sira', inplace=True, axis=1)
df.drop('sira2', inplace=True, axis=1)
df.drop('noname', inplace=True, axis=1)
df = df.sample(frac=1 , random_state=42).reset_index(drop=True)


x = df['TEXT_IN_ENGLISH']
y = df['label']

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)
y = df['label']

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=42)

notr_text = "you are so awful"
new_text3_vector = vectorizer.transform([notr_text])


nb_model = MultinomialNB()
nb_model.fit(x_train , y_train)
nb_predictions = nb_model.predict(x_test)
nb_accuracy = accuracy_score(y_test , nb_predictions)
nb_f1_score = f1_score(y_test , nb_predictions , average=None)
print("Naive Bayes Accuracy: " , nb_accuracy)
print("Naive Bayes f1 Score: " , nb_f1_score)
prediction_nb_new_text = nb_model.predict(new_text3_vector)
if prediction_nb_new_text == 'negative':
    print("Girdiginiz metin , negatif bir sentiment içeriyor.")
elif prediction_nb_new_text == 'positive':
    print("Girdiginiz metin , pozitif bir sentiment içeriyor.")
else:
    print("Girdiginiz metin , nötr bir sentiment içeriyor.")