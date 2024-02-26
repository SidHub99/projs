import numpy as np
import json
import random
import nltk
import pandas as pd
import pickle
import os
import tensorflow as tf 
from keras.models import Sequential
from keras.models import load_model
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Embedding, Flatten,Input,Dropout,LSTM,GRU,GlobalAveragePooling1D, Conv1D, MaxPool1D
from keras.callbacks import TensorBoard,EarlyStopping
import joblib
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
dataset= json.loads(open('Intent.json').read())

maxlen= 500


words=[]
sequences=[]
documents=[]
classes=[]
ignoreletters=['?','!','.',',','-','_']

def processing_json_dataset(dataset):
  tags = []
  inputs = []
  responses={}
  for intent in dataset['intents']:
    responses[intent['intent']]=intent['responses']
    for lines in intent['text']:
      inputs.append(lines)
      tags.append(intent['intent'])
  return [tags, inputs, responses]

[tags, inputs, responses] = processing_json_dataset(dataset)
dataset = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

for text in dataset['inputs']:
  dataset['inputs']= dataset['inputs'].str.lower()

# print(dataset.head())

tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(dataset['inputs'])
sequences=tokenizer.texts_to_sequences(dataset['inputs'])

padded_sequences=pad_sequences(sequences)
word_index=tokenizer.word_index
classes=[]
words=[]
for tag in dataset['tags']:
  if tag not in classes:
    classes.append(tag)

for inp in dataset['inputs']:
  wordlist=word_tokenize(inp)
  words.extend(wordlist)

word=[lemmatizer.lemmatize(word) for word in words if word not in ignoreletters]
words=sorted(set(word))
classes=sorted(set(classes))

pickle.dump(words,open('my_words.pkl','wb'))
pickle.dump(classes,open('my_classes.pkl','wb'))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labels=le.fit_transform(dataset['tags'])
le_dump=joblib.dump(le,'le.pkl')
label_dump=joblib.dump(labels,'labels_dump.pkl')
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
x_train,x_val,y_train,y_val=train_test_split(padded_sequences,labels,test_size=0.2)

model=Sequential()
model.add(Input(shape= padded_sequences.shape[1]))
model.add(Embedding(maxwords,embedding_dim,input_length=maxlen))
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', kernel_initializer='glorot_normal', bias_regularizer=tf.keras.regularizers.L2(0.0001), activity_regularizer=tf.keras.regularizers.L2(0.0001)))
model.add(Dropout(0.6))
model.add(LSTM(32,dropout=0.6,return_sequences=True))
model.add(LSTM(16,dropout=0.3,return_sequences=False))
model.add(Dense(128,activation='relu',activity_regularizer=tf.keras.regularizers.L2(0.0001)))
model.add(Dropout(0.3))
model.add(Dense(22,activation='softmax',activity_regularizer=tf.keras.regularizers.L2(0.0001)))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

early_stopping=EarlyStopping(monitor='loss',patience=400,mode='min',restore_best_weights=True)
history_training = model.fit(x_train,y_train,validation_data=[x_val,y_val],epochs=2000, batch_size=64, callbacks=[early_stopping])

model.save_weights("glove_chat.h5")
model.save('transfer_chatbot.h5',history_training)


# print('words',words.shape)
# print('words[0]',words.shape[0])
import matplotlib.pyplot as plt

plt.plot(history_training.history['loss'], label='Train')
plt.plot(history_training.history['val_loss'], label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# import random
# import string
# def generate_answer(query):
#   texts = []
#   pred_input = query
#   pred_input = [letters.lower() for letters in pred_input if letters not in string.punctuation]
#   pred_input = ''.join(pred_input)
#   texts.append(pred_input)
#   pred_input = tokenizer.texts_to_sequences(texts)
#   pred_input = np.array(pred_input).reshape(-1)
#   pred_input = pad_sequences([pred_input],9)
#   output = model.predict(pred_input)
#   output = output.argmax()
#   # if output not in le_dump:
#   #    print('Dont know what you are saying')
#   # else:
#   response_tag = le.inverse_transform([output])[0]
#   return random.choice(responses[response_tag])

# list_que = ["hello", "i am kaled","what is my name?",
#             "what is your name?", "tell me please, what is your name?"]
# for i in list_que:
#   print("you: {}".format(i))
#   res_tag = generate_answer(i)
#   print(res_tag)  