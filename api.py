from fastapi import FastAPI, UploadFile
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
import shutil

app = FastAPI()

model = None

def get_melspectrogram_fixed_length(file_path, n_mels=128, max_length=216):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        if spectrogram_db.shape[1] < max_length:
            pad_width = max_length - spectrogram_db.shape[1]
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spectrogram_db = spectrogram_db[:, :max_length]

        return spectrogram_db
    except Exception as e:
        print(f"Erreur avec {file_path}: {e}")
        return None

@app.on_event("startup")
def load():
    global model
    model_path = 'K13_emotion_model.h5'
    model = load_model(model_path)

@app.get("/")
def hello():
    return {"message": "Bienvenue sur l'API K13 🎲 !"}

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        file_path = os.path.join('/tmp', file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extraire le spectrogramme du fichier audio
        spec = get_melspectrogram_fixed_length(file_path)
        if spec is not None:
            spec = np.expand_dims(spec, axis=0)  # (1, 128, max_length)
            spec = np.expand_dims(spec, axis=-1)  # (1, 128, max_length, 1)

            # Prédire avec le modèle
            prediction = model.predict(spec)
            predicted_class = np.argmax(prediction, axis=1)[0]

            emotions = ["Neutre", "Calme", "Heureux", "Triste", 
                        "En colère", "Peur", "Dégoûté", "Surpris"]

            return {'Prediction': emotions[predicted_class]}
        else:
            return {"error": "Impossible de traiter le fichier audio."}
    finally:
        # Supprimer le fichier après traitement
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)