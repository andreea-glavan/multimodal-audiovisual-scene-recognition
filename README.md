# multimodal-indoor-scene-rec

If you find this material useful in your research, please cite:

@article{xxx, <br>
Author = {A. Glavan and E. Talavera}, <br>
Title = {InstaIndoor and Multi-modal Deep Learning for Indoor Scene Recognition}, <br>
journal  = {Neural Computing with Applications}, <br>
Year = {2021} <br>
}

This repository contains the source code for the multimodal RNN pipeline for scene recognition in videos. Moreover, we include the extracted multimodal features of the two novel datasets: namely, InstaIndoor and YouTubeIndoor.

The RNN pipeline utilizes joint fusion of visual (CNN processed frames) and text (speech transcribed from audio) by obtaining global descriptors using a ConvLSTM and a LSTM, respectively. This pipeline was applied to the two dataset with accuracy values of over 70%.

The two proposed datasets envelop 9 indoor scene classes each, collected through videos from social media. The videos were selected and filtered based on user annotation (hashtags, descriptions, etc.), but also based on manual filtering for quality control. InstaIndoor contains over 3000 videos collected from Instagram, whereas YouTubeIndoor consists of 900 videos selected from the YouTube-8M dataset. The datasets are available in feature vector format in the directory Features > InstaIndoor/YouTube, organized as such: 

Labels: train/test values per class

Text: raw transcribed text per video

Visual: ImageNet and Places365 summed values per frame per video 

The train:test split used is 70:30 and seeded such that ordering is preserved throughout the feature descriptor files.
