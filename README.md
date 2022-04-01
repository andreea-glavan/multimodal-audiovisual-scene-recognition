# InstaIndoor and Multi-modal Deep Learning for Indoor Scene Recognition

This repository contains the source code for the multimodal RNN pipeline for scene recognition in videos. Moreover, we include the extracted multimodal features of the two novel datasets: namely, InstaIndoor and YouTubeIndoor.

Indoor scene recognition is a growing field with great potential for behaviour understanding, robot localization, and elderly monitoring, among others. In this study, we approach the task of scene recognition from a novel standpoint, using multi-modal learning and video data gathered from social media. The accessibility and variety of social media videos can provide realistic data for modern scene recognition techniques and applications. We propose a model based on fusion of transcribed speech to text and visual features, which is used for classification on a novel dataset of social media videos of indoor scenes named InstaIndoor. Our model achieves up to 70% accuracy and 0.7 F1-Score. Furthermore, we highlight the potential of our approach by benchmarking on a YouTube-8M subset of indoor scenes as well, where it achieves 74% accuracy and 0.74 F1-Score. We hope the contributions of this work pave the way to novel research in the challenging field of indoor scene recognition.

The two proposed datasets envelop 9 indoor scene classes each, collected through videos from social media. The videos were selected and filtered based on user annotation (hashtags, descriptions, etc.), but also based on manual filtering for quality control. InstaIndoor contains over 3000 videos collected from Instagram, whereas YouTubeIndoor consists of 900 videos selected from the YouTube-8M dataset. 

If you find this material useful in your research, please cite:

```bash
@article{glavan2021, 
Author = {A. Glavan and E. Talavera}, 
Title = {InstaIndoor and Multi-modal Deep Learning for Indoor Scene Recognition}, 
journal  = {Neural Computing and Applications}, 
Year = {2021} 
}
```

The multimodal pipeline we propose utilizes joint fusion of visual (CNN processed frames) and text (speech transcribed from audio) by obtaining global descriptors using a ConvLSTM and a LSTM, respectively. The pipeline 

![Proposed MultiModal Pipeline for Scene Classification in Videos](https://i.imgur.com/nWqUoZQ.png)

For a video summary of the methodology and findings of this publication, please check out:
[![YouTube Summary of MultiModal Indoor Scene Recognition](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](youtu.be/wSP9bvQi7so)

## Repository Contents

The repository contains two directories: /Code and /Features. In /Code, the Python source code is available for the pipeline variations depending on the fusion strategy employed. In /Features, the two datasets are available in feature vector format, organized at dataset level (InstaIndoor/YouTube). The features are organized as such: 

/Labels: ground truth values

/Text: raw transcribed text per video

/Visual: ImageNet and Places365 summed values per frame per video 

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

