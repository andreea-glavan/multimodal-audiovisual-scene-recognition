### this file is the main RNN pipeline
# allows for joint and late fusion
# early fusion code is in a separate file
# can handle any of the feature types
# for images, a frame vs cnn parameter must be set accordinglt

path = "/data/s3083691/InstaIndoor/"

import moviepy.editor as mp
from sklearn.model_selection import KFold
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
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
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

# instagram
classes = ["19 Cafe", "20 Bar", "22 ReadingRoom", "24 IndoorStadium", "25 Arcade", "27 Library", "28 Closet", "29 BeautySalon", "30 Aquarium"] # folder names
class_names = ["Cafe", "Bar", "Reading Room", "Indoor Stadium", "Arcade", "Library", "Closet", "Beauty Salon", "Aquarium"] # class names


# youtube
#classes = ["01 Kitchen", "02 Gym", "03 Office", "04 Library", "05 Supermarket", "06 Stadium",  "07 Garage", "08 Museum", "09 Aquarium"] 
#class_names = ["Kitchen", "Gym", "Office", "Library", "Supermarket", "Stadium", "Garage", "Museum", "Aquarium"]

# hw2
#classes = ["EXT-House", "EXT-Road", "INT-Bedroom", "INT-Car", "INT-Hotel", "INT-Kitchen", "INT-LivingRoom", "INT-Office", "INT-Restaurant", "INT-Shop"]
#class_names = ["EXT-House", "EXT-Road", "INT-Bedroom", "INT-Car", "INT-Hotel", "INT-Kitchen", "INT-LivingRoom", "INT-Office", "INT-Restaurant", "INT-Shop"]

data_path = "/data/pg-instvid/"


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
batches = 1
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
# visual_mode: "frames" or "cnn" - type of img input, raw frames or cnn features
# cnn_size: size of cnn feature vector (1000 imagenet or 365 places)
# fusion: "late" or "joint" - determines whetehr we fuse softmax features or global features respectively
def IG_Net(visual_mode, cnn_size, fusion):
    # textual modality layers
    text_in = Input(shape=[max_len])
    layer = Embedding(max_words, 50, mask_zero = True, input_length = max_len, trainable = False)(text_in)
    #layer = Dropout(0.1)(layer) #s
    layer_text = LSTM(512)(layer)

    # visual modality layers - process frames with convlstm or cnn
    if visual_mode == "frames":
      img_in = Input(shape=(img_seq_len, img_height, img_width, 3))
      layer = ConvLSTM2D(filters = 512, kernel_size = (3, 3), return_sequences = False, data_format = "channels_last",
                input_shape = (img_seq_len, img_height, img_width, 3))(img_in)
      layer_img = Flatten()(layer)
    elif visual_mode == "cnn":
      img_in = Input(shape=[cnn_size])
      layer_img = Dense(512)(img_in)

    # generate final vectors based on fusion type (global vs softmax)
    if fusion == "joint":
      text_out = layer_text
      img_out = layer_img
    elif fusion == "late":
      # text
      text_out = Dense(num_classes, name='FC1')(layer_text) 
      # visual
      layer = Dense(256, activation="relu")(layer_img)
      img_out = Dense(num_classes, activation = "softmax")(layer)

    # fusion and final layers
    concat_inp = concatenate([text_out, img_out])
    z = Dense(256, activation='relu')(concat_inp)
    z = Dropout(0.3)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = concat_inp
    output = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs = [text_in, img_in], outputs = output) 

    return model


### load all data, saved as pickle files
# load text features
inf = open(path + 'text_e1_train', 'rb')
text_train = pickle.load(inf)
text_train = np.asarray(text_train)
inf.close()

inf = open(path + 'text_e1_test', 'rb')
text_test = pickle.load(inf)
text_test = np.asarray(text_test)
inf.close()

# load visual features
inf = open(path + 'img' + str(img_height) + '_train_' + str(img_seq_len), 'rb')
img_train = pickle.load(inf)
img_train = np.asarray(img_train)
inf.close()

inf = open(path + 'img' + str(img_height) + '_test_' + str(img_seq_len), 'rb')
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


# init model
# MUST specify img_type parameter based on visual feature unused
# possible values: frames, cnn
img_type = "frames" # frames or cnn
size = len(img_train[0]) # unused unless cnn
fusion = "joint"

model = IG_Net(img_type, size, fusion)

# init model features
opt = keras.optimizers.Adam(lr=lr)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ["accuracy"])

earlystop = EarlyStopping(patience=10)
callbacks = [earlystop]
 
# training
_ = model.fit(x = [text_train, img_train], y = train_labels, epochs = num_epochs, batch_size = batches,
                    shuffle = True, validation_split = 0.2, callbacks = callbacks)


# evaluate results on test set
res = model.predict([text_test, img_test], batch_size=1)

# compute quantitative metrics
test_labels = np.argmax(test_labels, axis = 1)

res = np.argmax(res, axis = 1)

print(classification_report(test_labels, res))
