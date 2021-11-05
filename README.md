# InstaIndoor and Multi-modal Deep Learning for Indoor Scene Recognition

This repository contains the source code for the multimodal RNN pipeline for scene recognition in videos. Moreover, we include the extracted multimodal features of the two novel datasets: namely, InstaIndoor and YouTubeIndoor.

The RNN pipeline utilizes joint fusion of visual (CNN processed frames) and text (speech transcribed from audio) by obtaining global descriptors using a ConvLSTM and a LSTM, respectively. This pipeline was applied to the two dataset with accuracy values of over 70%.

The two proposed datasets envelop 9 indoor scene classes each, collected through videos from social media. The videos were selected and filtered based on user annotation (hashtags, descriptions, etc.), but also based on manual filtering for quality control. InstaIndoor contains over 3000 videos collected from Instagram, whereas YouTubeIndoor consists of 900 videos selected from the YouTube-8M dataset. 

If you find this material useful in your research, please cite:

```bash
@article{xxx, 
Author = {A. Glavan and E. Talavera}, 
Title = {InstaIndoor and Multi-modal Deep Learning for Indoor Scene Recognition}, 
journal  = {Neural Computing with Applications}, 
Year = {2021} 
}
```


![Proposed MultiModal Pipeline for Scene Classification in Videos](https://i.imgur.com/nWqUoZQ.png)

## Repository Contents

The datasets are available in feature vector format in the directory Features > InstaIndoor/YouTube, organized as such: 

Labels: train/test values per class

Text: raw transcribed text per video

Visual: ImageNet and Places365 summed values per frame per video 

The train:test split used is 70:30 and seeded such that ordering is preserved throughout the feature descriptor files.



## Loading the Datasets

The data features are encoded using the Python library pickle. To load a specific file, use the following code:

```bash
import pickle
import numpy as np

path = '/Features/InstaIndoor/Labels/labels_test'

inf = open(path, 'rb')
test_labels = pickle.load(inf)
test_labels = np.asarray(test_labels)
inf.close()
```

## Try the MultiModal Pipeline

The pipeline source code is available in the /Code directory. Two files, depending on the type of fusion used, are available. Each of the files builds and trains the models from scratch, with the dataset used being specified within the source code. This dataset can be changed according to preference. To train and evaluate our pipeline, please run the following code:

```bash
python early_fusion.py
```

