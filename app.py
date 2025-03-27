from flask import Flask, request, jsonify
import joblib
import io
import soundfile as sf
import librosa
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('model.pkl')
n_mfcc = 40
n_fft = 1024  # setting the FFT size to 1024
hop_length = 10*16 # 25ms*16khz samples has been taken
win_length = 25*16 #25ms*16khz samples has been taken for window length
window = 'hann' #hann window used
n_chroma=12
n_mels=128
n_bands=7 #we are extracting the 7 features out of the spectral contrast
fmin=100
bins_per_ocatve=12
def extract_features(file_path):
    try:
        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',n_mels=n_mels).T,axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr,n_fft=n_fft,
                                                      hop_length=hop_length, win_length=win_length,
                                                      n_bands=n_bands, fmin=fmin).T,axis=0)
        tonnetz =np.mean(librosa.feature.tonnetz(y=y, sr=sr).T,axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        # print(shape(features))
        return features
    except:
        print("Error: Exception occurred in feature extraction")
        return None

@app.route('/')
def home():
    return "Infant Cry Classification API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive audio file from ESP32
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        print("File : ",audio_file)

        # Read the audio file into memory
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Save it temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        # ✅ Use existing extract_features function from GitHub repo
        feature = extract_features("temp_audio.wav")

        features = []
        features.append(feature)
        # Reshape for model input
        
        features = np.array(features)
        print("Given : ",features)
        input_features = features.reshape(1, -1)
        # Make prediction
        prediction = model.predict(np.array(input_features))

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
