import streamlit as st

# Streamlit app
st.set_page_config(page_title="Audio Sentiment Analysis", page_icon="🎵")

import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_analysis_model.joblib')

model = load_model()

# Function to extract features from audio
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return np.hstack((np.mean(mfcc, axis=1),
                      np.mean(spectral_centroid, axis=1),
                      np.mean(chroma, axis=1)))

# Function to plot waveform
def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    return fig


st.title('🎵 Audio Sentiment Analysis')

st.markdown("""
This app predicts the sentiment of an uploaded audio file.
Upload a WAV file and click 'Predict Sentiment' to see the results.
""")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button('Predict Sentiment'):
        with st.spinner('Analyzing audio...'):
            try:
                # Load the audio file
                y, sr = librosa.load(uploaded_file)

                # Plot waveform
                st.pyplot(plot_waveform(y, sr))

                # Extract features
                features = extract_features(y, sr)

                # Reshape features for prediction
                features = features.reshape(1, -1)

                # Make prediction
                prediction = model.predict(features)[0]

                # Display result
                st.success(f'Predicted Sentiment: {prediction}')

                # Display confidence scores
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0]
                    st.write('Confidence Scores:')
                    for sentiment, prob in zip(model.classes_, probabilities):
                        st.progress(prob)
                        st.write(f'{sentiment}: {prob:.2f}')

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.sidebar.header("About")
st.sidebar.info("""
This app uses machine learning to predict the sentiment of audio files.
It extracts features using librosa and uses a pre-trained model for prediction.
""")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a WAV file using the file uploader.
2. Click the 'Predict Sentiment' button.
3. View the predicted sentiment and confidence scores.
""")