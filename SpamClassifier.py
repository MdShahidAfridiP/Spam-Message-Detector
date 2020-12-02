# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:52:19 2020

@author: Shahid
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import re
from tensorflow.keras.callbacks import EarlyStopping
import pickle

dataset=pd.read_csv("SMSSpamCollection",sep="\t",names=["detect","review"])
x=dataset.iloc[:,-1]
y=dataset.iloc[:,0]

corpus=[]
wt=WordNetLemmatizer()
for i in range(len(x)):
    msg=re.sub("[^a-zA-Z]"," ",x[i])
    msg=msg.lower()
    msg=msg.split()
    msg=[wt.lemmatize(word) for word in msg if word not in set(stopwords.words("english"))]
    msg=" ".join(msg)
    corpus.append(msg)

tfidf=TfidfVectorizer(max_features=5000)
x=tfidf.fit_transform(corpus).toarray()

y=np.where(y=="spam",1,0)   

pickle.dump(tfidf,open("transform.pkl","wb"))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from imblearn.combine import SMOTETomek
sm=SMOTETomek(random_state=42)
xtrain_sm,ytrain_sm=sm.fit_sample(xtrain,ytrain)

earlystop=EarlyStopping(monitor="val_loss",patience=10)

classifier=Sequential()
classifier.add(Dense(units=2500,kernel_initializer="he_uniform",activation="relu",input_dim=5000))
classifier.add(Dense(units=2500,kernel_initializer="he_uniform",activation="relu"))
classifier.add(Dense(units=1,kernel_initializer="he_uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(xtrain_sm,ytrain_sm,batch_size=100,epochs=100,validation_data=(xtest,ytest),callbacks=earlystop)

ypred=classifier.predict(xtest)

print(confusion_matrix(ytest,(ypred>0.5)))
print(accuracy_score(ytest,(ypred>0.5)))

classifier.save("spam_model.h5")

def message(msg):
    msg=re.sub("[^a-zA-Z]"," ",msg)
    msg=msg.lower()
    msg=msg.split()
    msg=[wt.lemmatize(word) for word in msg if word not in set(stopwords.words("english"))]
    msg=" ".join(msg)
    msg=tfidf.transform([msg]).toarray()
    result=classifier.predict([msg])
    print(result)
    if (result[0][0])>=0.5:
        print("spam")
    else:
        print("not spam")

message(input("enter your message : "))