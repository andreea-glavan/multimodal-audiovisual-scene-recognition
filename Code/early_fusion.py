import moviepy.editor as mp
import numpy as np
import sys
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import Model, load_model
from keras.layers import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import optimizers
import keras
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve, balanced_accuracy_score, accuracy_score
import pickle
from sklearn.utils import class_weight
import math
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import pickle
import cv2
import torchvision
from tqdm import tqdm
import os
import gensim 
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import KFold

path = "/Features/InstaIndoor/"

# instaindoor
classes = ["19 Cafe", "20 Bar", "22 ReadingRoom", "24 IndoorStadium", "25 Arcade", "27 Library", "28 Closet", "29 BeautySalon", "30 Aquarium"] # folder names
class_names = ["Cafe", "Bar", "Reading Room", "Indoor Stadium", "Arcade", "Library", "Closet", "Beauty Salon", "Aquarium"] # class names

# youtube 8m
#classes = ["01 Kitchen", "02 Gym", "03 Office", "04 Library", "05 Supermarket", "06 Stadium",  "07 Garage"] #, "08 Museum", "09 Aquarium"] 
#class_names = ["Kitchen", "Gym", "Office", "Library", "Supermarket", "Stadium", "Garage"] #, "Museum", "Aquarium"]

### global parameters
# image features
img_height, img_width = 64, 64
img_seq_len = 10
max_words = 10000

# text features
max_len = 100

# network features
num_epochs = 20
lr = 0.001
batches = 8
num_classes = len(classes)

# helper function. plots the confusion matrix based on classification results
def plot_cm(labels, res, class_names):
    cm = confusion_matrix(labels, res)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]

            annot[i, j] = '%.1f%%' % (p)
            annot[i, j] = annot[i, j].replace('%', '')
            print(annot[i,j])

    cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig, ax = plt.subplots(figsize=(9,9))
    sns.heatmap(cm, annot=annot, cmap = "BuPu", fmt='', ax=ax)
    plt.show()

# multimodal network
# input: 
# cnn_size - size of the feature vector per img (1000 for imagenet or 365 for places)
# embedding_len - size of the text feature vector (100 for word2vec embeddings or 768 for sentencebert)
def IG_Net(cnn_size, embedding_len):
    text_in = Input(shape=([embedding_len]))
    img_in = Input(shape=[cnn_size])

    concat_inp = concatenate([text_in, img_in])

    z = Dense(512, activation='relu')(concat_inp)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.1)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.1)(z)
    output = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs = [text_in, img_in], outputs = output) 

    return model

### load all data, saved as pickle files
# load text features
inf = open(path + 'text_count_train', 'rb')
text_train = pickle.load(inf)
inf.close()

inf = open(path + 'text_count_test', 'rb')
text_test = pickle.load(inf)
inf.close()

# load visual features
inf = open(path + 'in_train_sum' , 'rb')
img_train = pickle.load(inf)
img_train = np.asarray(img_train)
inf.close()

inf = open(path + 'in_test_sum', 'rb')
img_test = pickle.load(inf)
img_test = np.asarray(img_test)
inf.close()

# load labels
inf = open(path + 'labels_train', 'rb')
train_labels = pickle.load(inf)
train_labels = np.asarray(train_labels)
inf.close()

inf = open(path + 'labels_test', 'rb')
test_labels = pickle.load(inf)
test_labels = np.asarray(test_labels)
inf.close()

# determine param based on features
cnn_size = len(img_train[0])
embedding_len = len(text_train[0])

# init model
model = IG_Net(cnn_size, embedding_len)

# init model features
opt = keras.optimizers.Adam(lr=lr)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ["accuracy"])

earlystop = EarlyStopping(patience=7)
callbacks = [earlystop]

_ = model.fit(x = [text_train[:len(img_train)], img_train], y = train_labels, epochs = num_epochs, batch_size = batches,
                    shuffle = True, validation_split = 0.2, callbacks = callbacks)

# predict on test set
res = model.predict([text_test, img_test], batch_size=1)

# compute quantitative metrics
test_labels = np.argmax(test_labels, axis = 1)

res = np.argmax(res, axis = 1)

print(classification_report(test_labels, res))
