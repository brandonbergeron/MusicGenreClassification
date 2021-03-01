import streamlit as st
import numpy as np
import librosa, librosa.display
import io
from PIL import Image
from keras.models import load_model
from scipy.io.wavfile import write
import scipy.signal as sps
import sounddevice as sd

signal = None
RECORD_SR = 48_000
DESIRED_SR = 22_050

model = load_model('./models/harm_perc_1.h5')

labels = ['pop',
          'metal',
          'disco',
          'blues',
          'reggae',
          'classical',
          'rock',
          'hiphop',
          'country',
          'jazz']

def get_mfcc(signal):
    mfcc = librosa.feature.mfcc(signal, n_mfcc=13).T
    
    mfcc = mfcc.reshape(-1, mfcc.shape[0], mfcc.shape[1], 1)
    
    return mfcc

def get_harm_perc(signal):
    #signal, _ = librosa.load('output.wav')
    signal_harmonic, signal_percussive = librosa.effects.hpss(signal)
    
    mfcc_harmonic = librosa.feature.mfcc(signal_harmonic, n_mfcc=13).T
    mfcc_harmonic = mfcc_harmonic.reshape(1, mfcc_harmonic.shape[0], mfcc_harmonic.shape[1], 1)
    
    mfcc_percussive = librosa.feature.mfcc(signal_percussive, n_mfcc=13).T
    mfcc_percussive = mfcc_percussive.reshape(1, mfcc_percussive.shape[0], mfcc_percussive.shape[1])


    return [mfcc_harmonic, mfcc_percussive]

def main():
    title = "Music Genre Classifier"
    st.title(title)

    if st.button('Record'):
        with st.spinner('Recording...'):
            signal = sd.rec(int(3*48000), samplerate=48000, channels=1, blocking=True, dtype='float64')
            sd.wait()
            signal = signal.reshape(signal.shape[0])
            # st.write(signal.shape)
            # st.write(signal)

            new_num_samples = round(len(signal) * float(DESIRED_SR) / RECORD_SR)
            signal = sps.resample(signal, new_num_samples)
            
            # st.write(signal.shape)
            # st.write(type(signal))
            # st.write(signal)

            st.experimental_set_query_params(my_saved_result=signal)
        
        st.success("Recording completed")

    app_state = st.experimental_get_query_params()  

    if st.button('Play'):
        try:
            signal = np.array(app_state["my_saved_result"], dtype='float')
            temp_file = io.BytesIO()  
            write(temp_file, 22050, signal)  
            st.audio(temp_file, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        with st.spinner("Thinking..."):
            signal = np.array(app_state["my_saved_result"], dtype='float')
            # st.write(type(signal))
            # st.write(signal.shape)
            # st.write(signal)
            pred = model.predict(get_harm_perc(signal))
        st.success("Classification completed")
        labels[np.argmax(pred)]

if __name__ == '__main__':
    main()