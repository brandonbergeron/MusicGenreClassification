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

  

## Data Collection

For this project, I used the GTZAN Genre Collection dataset which consists of 100 30-second examples from 10 different genres of music each. This is available for direct download from the link provided [here](http://marsyas.info/downloads/datasets.html) and instructions on where to store it in the repo can be found in the first [notebook](https://github.com/brandonbergeron/MusicGenreClassification/blob/master/code/01-EDA.ipynb)


## EDA/Feature Extraction

Fortunately the dataset for this project didn't require any cleaning. For EDA and data exploration I used [Librosa](https://librosa.org/doc/latest/index.html#) extensively, which has great capability for visualising extracted features. After reviewing similar projects in the field the general consensus seems to be that MFCCs (Mel-Frequency Cepstral Coefficients) are more effective and efficient for audio classification than spectrograms, so I decided to explore these further. I extracted 13, 20, and 30 coefficients from the audio in 10, 6, and 3 second segments creating 9 datasets.

Later, using the same 3s segments and 13 MFCCs, I went back and separated the audio into it's percussive and harmonic elements using Librosa. This proved effective in a multi-input layer model. 

It should be noted that most of my understanding of audio signal processing, as well as the methods for extracting features from audio come from Valerio Velardo's [Sound of AI](https://github.com/musikalkemist/AudioSignalProcessingForML) GitHub, which is accompanied by a thorough [YouTube](https://www.youtube.com/c/ValerioVelardoTheSoundofAI) series explaining each notebook. A valuable resource for anyone starting out in music and machine learning.

## Modeling

I fit a variety of models to the data including Convolutional Networks (CNN), LSTM networks, Convolutional/LSTM, Hidden Markov Models, and finally a multi-input-layer network using both the CNN and ConvLSTM side-by-side.

First, I fit a basic CNN to all 9 datasets created by [this notebook](https://github.com/brandonbergeron/MusicGenreClassification/blob/master/code/02-Extract_mfcc.ipynb) to asses how many coefficients/what length of audio segment might perform best. All of the examples with 13 MFCCs performed well, and although the 3-second segments performed marginally better than 6-seconds, I decided to proceed with 6-second segments because it provides a more complete representation of the tracks. I also used the dataset of 6-second/13-MFCC clips for the remainder of my modeling as less coefficients also reduced the size of the dataset considerably.

All of the Neural Networks performed well, and in an effort to boost their performance I combined the ConvLSTM and CNN side-by-side and see if there was something to be gained from both. To do this, I used the Keras Funcional API for multi-input capability. This model performed slightly better, and spread the errors more evenly so there wasn't just one genre pulling the performance down. I fit 3 different models using this method. The first model recieved the same MFCCs for both the CNN and ConvLSTM networks. This model showed slight increase in performance. 

For the final 2 Neural Networks, I used Librosa's "harmonic-percussive source separation" function to separate the audio signal into a harmonic and percussive signal. Audio examples of what these different signals sound like can be found in the [EDA](https://github.com/brandonbergeron/MusicGenreClassification/blob/master/code/01-EDA.ipynb) notebook. The harmonic signal was fed through the CNN side, and the percussive signal went through the ConvLSTM side of the network. The second of these utilized some custom data augmentation in attempt to boost performance of the web-app, which improved performance only slightly, making it a better model for the streamlit app. 

## Data Limitations and Constraints

As with any Data Science problem, more data is usually better. While the models created performed well, there were obvious limitations when testing them on more outside data. With only 100 songs from each genre, the representation of variety within each genre is limited. Jazz music with guitar tended to be classified as Blues, and Rock music with a lighter instrumental interlude often classified as Reggae or country. These are subtle differences that may not be discernable to even an experienced listener when presented with just a 6 second audio clip. With more data, there would be potential for the models to be fit to longer segments of audio without performance suffering. 

For the streamlit app, the recording quality of a laptop microphone paired with the downsampling and compression it takes to shape it correctly for making predictions is just not a good enough representation of the audio.

Another limitation when classifying music by genre is that many artists are influenced by different genres and often times make efforts to blur these lines in their own music. Whether it's a reggae break in a Led Zeppelin song, or an acoustic version of a rock song that sounds like country, this does not lend itself well to classification- which I will talk more about below in future work. 

## Future Work

I started experimenting with audio data augmentation including some basic pitch/time shifting as well as the addition of light white noise, but was not able to see a large boost in performance on the validation data. I saw marginal increase in performance of the streamlit app, but I would like to spend more time researching what types/methods of audio augmentation are effective for music classification, and see if I can implement them with any more success. 

In addition, there are a few aspects of the streamlit app I developed that need work. First, I would like to add the ability for the user to record indefinitely. Second, the recording process introduces quite a bit of white noise and other artifacts into the audio due to the downsampling from the laptop mic's sampling rate to the sampling rate of the dataset. I feel that data augmentation should improve the model's ability to predict on this altered sound, but my limited efforts have not been successful thus far. More effective data augmentation may solve this issue as well.

Lastly, this dataset is a valuable resource because of the fact that the music in it is popular music from each genre. So often it is the case when using resources like Free Music Archive that the music is just not representative of the actual genres. However, I think the scope of music for some genres in the GTZAN collection is too wide. Especially today, when songs have more than one genre tag, training a model on narrower definitions of each genre could prove to be useful.


## Final Conclusions and Summary
