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


app = FastAPI()
gecici_db = []
class TextData(BaseModel):
    text: str

def text_lowercase(TextData):
    return TextData.lower()

def remove_numbers(TextData):
    result = re.sub(r'/d+','',TextData)
    return result

def remove_punctuations(text):
    for punctuations in string.punctuation:
        text = str(text).replace(punctuations, '')
    return text
def tokenize_word_stopwords(text):
    stop_words = stopwords.words('english')
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


@app.get("/SentenceList")
async def get_sentence():
    return gecici_db


@app.post("/predict")
def predict_sentiment(text_data: TextData):
    new_text = text_data.text
    df = pd.read_csv('all-data7.csv')
    df.columns = ["noname","sira", "sira2", "TEXT_IN_ENGLISH", "label"]
    df.drop('sira', inplace=True, axis=1)
    df.drop('sira2', inplace=True, axis=1)
    df.drop('noname', inplace=True, axis=1)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['Re_Punc'] = df['TEXT_IN_ENGLISH'].apply(remove_punctuations)
    df['Re_Punc'] = df['Re_Punc'].str.lower()
    df.drop('TEXT_IN_ENGLISH', inplace=True, axis=1)
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(df['Re_Punc'])
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    nb_predictions = nb_model.predict(x_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    nb_f1_score = f1_score(y_test, nb_predictions, average=None)
    new_text = remove_punctuations(new_text)
    print(text_lowercase(new_text))
    print(remove_punctuations(new_text))
    new_text_vector = vectorizer.transform([new_text])
    prediction_nb_new_text = nb_model.predict(new_text_vector)
    gecici_db.append(new_text_vector)
    print(gecici_db)
    if prediction_nb_new_text  == 'negative':
        sentiment = "Girdiginiz metin , negatif bir sentiment içeriyor."
    elif prediction_nb_new_text == 'positive':
        sentiment = "Girdiginiz metin , pozitif bir sentiment içeriyor."
    else:
        sentiment = "Girdiginiz metin , nötr bir sentiment içeriyor."
    return {"sentiment : " + sentiment}












df=pd.read_csv('all-data7.csv')
df.columns = ["noname","sira","sira2","TEXT_IN_ENGLISH","label"]
df.drop('sira', inplace=True, axis=1)
df.drop('sira2', inplace=True, axis=1)
df.drop('noname', inplace=True, axis=1)
df = df.sample(frac=1 , random_state=42).reset_index(drop=True)
print(df)
#On İşleme
df['Re_Punc'] = df['TEXT_IN_ENGLISH'].apply(remove_punctuations)

df['Re_Punc'] = df['Re_Punc'].str.lower()
df.drop('TEXT_IN_ENGLISH', inplace = True , axis = 1)
print(df)
y = df['label']
filtered_texts = [tokenize_word_stopwords(text) for text in df['Re_Punc']]
filtered_data = pd.DataFrame({'Re_Punc' : filtered_texts,'label' : y})
print(filtered_data)

x = filtered_data['Re_Punc']
y = filtered_data['label']
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)
y = df['label']

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=42)

notr_text = "you are so awful"
new_text3_vector = vectorizer.transform([notr_text])
#naive Bayes
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


#2.Logistic Regression

lr_model =LogisticRegression(max_iter=1000)
lr_model.fit(x_train , y_train)
lr_predictions = lr_model.predict(x_test)
lr_accuracy = accuracy_score(y_test , lr_predictions)
lr_f1_score = f1_score(y_test , lr_predictions, average=None)
prediction_lr_new_text = lr_model.predict(new_text3_vector)
print("Logistic Regression Accuracy :" , lr_accuracy)
print("Logistic Regression f1 Score : " , lr_f1_score)
if prediction_nb_new_text == 'negative':
    print("Girdiginiz metin , negatif bir sentiment içeriyor.")
elif prediction_nb_new_text == 'positive':
    print("Girdiginiz metin , pozitif bir sentiment içeriyor.")
else:
    print("Girdiginiz metin , nötr bir sentiment içeriyor.")


# 3.Support Vector Machine (SVM)

# svm_model = SVC()
# svm_model.fit(x_train , y_train)
# svm_predictions = svm_model.predict(x_test)
# svm_accuracy = accuracy_score(y_test , svm_predictions)
# svm_f1_score = f1_score(y_test , svm_predictions , average=None)
# prediction_svm_new_text = svm_model.predict(new_text3_vector)
# print("SVM Accuracy :" , svm_accuracy)
# print("SVM F1 Score :" , svm_f1_score)
# if prediction_nb_new_text == 'negatif':
#     print("Girdiginiz metin , negatif bir sentiment içeriyor.")
# elif prediction_nb_new_text == 'pozitif':
#     print("Girdiginiz metin , pozitif bir sentiment içeriyor.")
# else:
#     print("Girdiginiz metin , nötr bir sentiment içeriyor.")
#
# #4.Random Forest
#
# rf_model = RandomForestClassifier()
# rf_model.fit(x_train,y_train)
# rf_predictions = rf_model.predict(x_test)
# rf_accuracy = accuracy_score(y_test , rf_predictions)
# rf_f1_score = f1_score(y_test , rf_predictions, average=None)
# prediction_rf_new_text = rf_model.predict(new_text3_vector)
# print("Random Forest Accuracy: ", rf_accuracy)
# print("Random Forest f1 Score: ", rf_f1_score)
# if prediction_nb_new_text == 'negative':
#     print("Girdiginiz metin , negatif bir sentiment içeriyor.")
# elif prediction_nb_new_text == 'positive':
#     print("Girdiginiz metin , pozitif bir sentiment içeriyor.")
# else:
#     print("Girdiginiz metin , nötr bir sentiment içeriyor.")

