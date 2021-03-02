# Music Genre Classification

Brandon Bergeron | DSI-113020 | 03.01.2021

## Contents:

#### - [Problem Statement](https://github.com/Rameshbabupv/dsi1130-project-5#problem-statement)
#### - [Project Directory](https://github.com/Rameshbabupv/dsi1130-project-5#project-directory)
#### - [Executive Summary](https://github.com/Rameshbabupv/dsi1130-project-5#executive-summary)
#### - [Data Collection](https://github.com/Rameshbabupv/dsi1130-project-5#data-collection)
#### - [Data Cleaning and EDA](https://github.com/Rameshbabupv/dsi1130-project-5#data-cleaning-and-eda)
#### - [Modeling](https://github.com/Rameshbabupv/dsi1130-project-5#modeling)
#### - [Data Limitations and Constraints](https://github.com/Rameshbabupv/dsi1130-project-5#data-limitations-and-constraints)
#### - [Future Work](https://github.com/Rameshbabupv/dsi1130-project-5#future-work)
#### - [Final Conclusions and Summary](https://github.com/Rameshbabupv/dsi1130-project-5#final-conclusions-and-summary)

## Problem Statement

Combining the fields of Music and Machine Learning has been something I have been interested in for a while, and has great application in areas such as music genre classification, tagging, and even music generation. Using the original [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html), the objective of this project was to learn more about audio signal processing and try my hand at a popular music classification problem. 


## Project Directory
```
|__ code
|   |__ 01-EDA.ipynb
|   |__ 02-Extract_mfcc.ipynb
|   |__ 03-Extract_harm_perc.ipynb
|   |__ 04-CNN_mfcc.ipynb
|   |__ 05-RNN_mfcc.ipynb
|   |__ 06-CRNN_mfcc.ipynb
|   |__ 07-HMM.ipynb
|   |__ 08-Keras_API.ipynb
|   |__ 09-Harm_Perc_API.ipynb
|   |__ 10-Audio_Augmentation.ipynb
|   |__ 11-Model_Augment.ipynb
|
|__ data
|   |__ mfcc_13_3s.json
|
|__ presentation
|   |--jazz_harmonic.wav
|   |--jazz_percussive.wav
|   |--model.png
|   |--piano_scale.jpg
|
|__ models
|   |-- first_model.h5
|   |-- harm_perc1.h5
|
|__ requirements.txt
|__ streamlit.py
|__ README.md
```

## Executive Summary

I took the popular GTZAN genre collection dataset and tried several different modeling strategies to  

## Data Collection

For this project, I used the GTZAN Genre Collection dataset which consists of 100 30-second examples from 10 different genres of music. This is available for direct download from the link provided [here](http://marsyas.info/downloads/datasets.html) and instructions on where to store it in the repo can be found in the first [notebook](https://github.com/brandonbergeron/MusicGenreClassification/blob/master/code/01-EDA.ipynb)


## EDA/Feature Extraction

Fortunately the dataset for this project didn't require any cleaning. For EDA and data exploration I used [Librosa] extensively, which has great capability for visualising extracted features. After reviewing similar projects in the field the general consensus seems to be that MFCCs (Mel-Frequency Cepstral Coefficients) are more effective and efficient for audio classification than spectrograms, so I decided to explore these further. I extracted 13, 20, and 30 coefficients from the audio in 10, 6, and 3 second segments creating 9 datasets.

Later, using the same 3s segments and 13 MFCCs, I went back and separated the audio into it's percussive and harmonic elements using Librosa. This proved effective in a multi-input layer model. 

It should be noted that most of my understanding of audio signal processing, as well as the methods for extracting features from audio come from Valerio Velardo's [Sound of AI]() GitHub, which is accompanied by a thorough [YouTube] series explaining each notebook. A valuable resource for anyone starting out in music and machine learning.

## Modeling

I fit a variety of models to my data including Convolutional Networks (CNN), LSTM networks, Conv/LSTM, Hidden Markov Models, and finally a multi-input-layer model using both the CNN and ConvLSTM side-by-side.

First, I fit a basic Convolutional Network to all 9 datasets created by [this notebook](notebook 2) to asses how many coefficients/what length of audio segment might perform best. All of the 3 second segments performed well, and 13 coefficients showed the least overfitting. I decided to use the dataset of 3-second/13-MFCC clips for the remainder of my modeling as less coefficients also reduced the size of the dataset considerably.

All of the Neural Networks performed well, and in an effort to boost their performance I decided to combine the ConvLSTM and CNN and see if there wasn't something to be gained from both.

## Data Limitations and Constraints

As with any Data Science problem, more data is usually better. While the models created performed well, there were obvious limitations when testing them on more outside data. With only 100 songs from each genre, the representation of variety within each genre is limited. Jazz music with guitar tended to be classified as Blues, and Rock music with a lighter instrumental interlude often classified as Reggae. These are subtle differences that may not be discernable to even a trained musician when presented with just a 3 second audio clip.

## Future Work




## Final Conclusions and Summary
