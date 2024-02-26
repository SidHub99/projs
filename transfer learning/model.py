import numpy as np
import json
import random
import nltk
import pickle
import os
import tensorflow as tf 
from keras.models import Sequential
from keras.models import load_model
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Embedding, Flatten,Input,Dropout,LSTM,GRU,GlobalAveragePooling1D, Conv1D, MaxPool1D
from keras.callbacks import TensorBoard,EarlyStopping

lemmatizer=WordNetLemmatizer()
intents= json.loads(open('Intent.json').read())

maxlen= 500


words=[]
sequences=[]
documents=[]
classes=[]
ignoreletters=['?','!','.',',','-','_']

tokenizer=Tokenizer(num_words=10000)
# for intent in intents['intents']:
#     for text in intent['text']:
#         wordslist= word_tokenize(text)
#         words.extend(wordslist)
#         for word in wordslist:
#             sequences.append(tokenizer.texts_to_sequences([word])[0])
#         documents.append((wordslist , intent['intent']))
#         if intent['intent'] not in classes:
#             classes.append(intent['intent'])

# print(words,classes)

text_data = [text for intent in intents['intents'] for text in intent['text']]
tokenizer.fit_on_texts(text_data)
sequences=tokenizer.texts_to_sequences(text_data)

for intent in intents['intents']:
    for clas in intent['intent']:
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

word_index=tokenizer.word_index
data=pad_sequences(sequences,maxlen=maxlen)
classes=np.asarray(classes)

# print(data.shape)
# print(classes.shape)
glove=(r'C:\Users\User\Downloads\gloves')
embedding_index={}
f=open(os.path.join(glove,'glove.6B.100d.txt'),encoding='utf-8')
for line in f:
    values=line.split()
    words=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embedding_index[words]=coefs
f.close

embedding_dim=100
maxwords= len(tokenizer.word_index) + 1 
embedding_matrix=np.zeros((maxwords,embedding_dim))
for word,i in word_index.items():
    if i < maxwords:
        embedding_vector=embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
# print(embedding_matrix)
# print(data.shape)
# print(data)

model=Sequential()
model.add(Input(shape= data.shape[1]))
model.add(Embedding(maxwords,embedding_dim,input_length=maxlen))
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', kernel_initializer='glorot_normal', bias_regularizer=tf.keras.regularizers.L2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001)))
model.add(Dropout(0.3))
model.add(LSTM(32,dropout=0.3,return_sequences=True))
model.add(LSTM(16,dropout=0.3,return_sequences=False))
model.add(Dense(128,activation='relu',activity_regularizer=tf.keras.regularizers.L2(0.0001)))
model.add(Dropout(0.6))
model.add(Dense(22,activation='softmax',activity_regularizer=tf.keras.regularizers.L2(0.0001)))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False
model.compile(loss='sparse_categrorical_crossentropy',optimizer='adam',metrics=['accuracy'])

early_stopping=EarlyStopping(monitor='loss',patience=400,mode='min',restore_best_weights=True)
history_training = model.fit(data,classes,epochs=2000, batch_size=64, callbacks=[early_stopping])

model.save_weights("glove_chat.h5")


