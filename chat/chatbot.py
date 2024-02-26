import numpy as np
import random
import json
import pickle
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

lemmatizer=WordNetLemmatizer()
intents=json.loads(open(r"C:\Users\User\OneDrive\Desktop\projects\New folder\intents.json").read())

words=[]
classes=[]
documents=[]
ignoreletters=['?',"!",".",","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist= word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

word=[lemmatizer.lemmatize(word)
       for word in words if word not in ignoreletters]
words=sorted(set(word))

classes=sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
training=[]
outputEmpty= [0] * len(classes)

for document in documents:
    bag=[]
    wordpatterns= document[0]
    wordpatterns=[lemmatizer.lemmatize(word.lower())
                  for word in wordpatterns
                  ]
    for word in words:
        bag.append(1) if word in wordpatterns else bag.append(0)

    outputrow= list(outputEmpty)
    outputrow[classes.index(document[1])] = 1
    training.append(bag + outputrow)

random.shuffle(training)
training = np.array(training)
trainx= training[:, :len(words)]
trainy= training[:, len(words):]
print(trainx.shape)
print(trainy.shape)
print(training[0])

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainx[0]),),activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainy[0]),activation='softmax'))

sgd=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(trainx),np.array(trainy),epochs=200,batch_size=5,verbose=1)

model.save('chatbot_model.h5',hist)
print('Done')
print(trainy)

model.summary