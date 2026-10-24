import os
import time
import datetime
import numpy as np
import cv2
import librosa
import sounddevice as sd
import soundfile as sf
import gradio as gr
import geocoder
from geopy.geocoders import Nominatim
from joblib import load
from tensorflow.keras.models import load_model
from twilio.rest import Client
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import subprocess
import joblib
import subprocess
import shutil
import time
import os


# ==== CONFIG 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Charger modèles ML
#svm_model = load(os.path.join(BASE_DIR, 'svm_model.joblib'))
#mpn_model = load_model(os.path.join(BASE_DIR, 'mpn_model.h5'))
#rf_model = load(os.path.join(BASE_DIR, 'random_forest_model.joblib'))
#knn_model = load(os.path.join(BASE_DIR, 'knn_model.joblib'))

# Chargement des modèles Keras 
mpn_model = load_model(os.path.join(BASE_DIR, 'mpn_model.keras'))
cnn_model = load_model(os.path.join(BASE_DIR, 'cnn_model.keras'))

#Chargement des modèles scikit-learn
svm_model = joblib.load(os.path.join(BASE_DIR, 'svm_model.joblib'))
linear_svc_model = joblib.load(os.path.join(BASE_DIR, 'linearsvc_model.joblib'))
rf_model = joblib.load(os.path.join(BASE_DIR, 'randomforest_model.joblib'))
et_model = joblib.load(os.path.join(BASE_DIR, 'extratrees_model.joblib'))
gb_model = joblib.load(os.path.join(BASE_DIR, 'gradientboosting_model.joblib'))
ada_model = joblib.load(os.path.join(BASE_DIR, 'adaboost_model.joblib'))
xgb_model = joblib.load(os.path.join(BASE_DIR, 'xgboost_model.joblib'))
lgb_model = joblib.load(os.path.join(BASE_DIR, 'lightgbm_model.joblib'))
knn_model = joblib.load(os.path.join(BASE_DIR, 'knn_model.joblib'))
dt_model = joblib.load(os.path.join(BASE_DIR, 'decisiontree_model.joblib'))
nb_model = joblib.load(os.path.join(BASE_DIR, 'naivebayes_model.joblib'))
lr_model = joblib.load(os.path.join(BASE_DIR, 'logisticregression_model.joblib'))
ridge_model = joblib.load(os.path.join(BASE_DIR, 'ridgeclassifier_model.joblib'))
perc_model = joblib.load(os.path.join(BASE_DIR, 'perceptron_model.joblib'))
mlp_model = joblib.load(os.path.join(BASE_DIR, 'mlpclassifier_model.joblib'))
lda_model = joblib.load(os.path.join(BASE_DIR, 'lda_model.joblib'))
qda_model = joblib.load(os.path.join(BASE_DIR, 'qda_model.joblib'))
sgd_model = joblib.load(os.path.join(BASE_DIR, 'sgdclassifier_model.joblib'))
bagging_model = joblib.load(os.path.join(BASE_DIR, 'baggingclassifier_model.joblib'))
#nusvc_model = joblib.load(os.path.join(BASE_DIR, 'nusvc_model.joblib'))
#calibrated_svc_model = joblib.load(os.path.join(BASE_DIR, 'calibratedsvc_model.joblib'))
#radius_neighbors_model = joblib.load(os.path.join(BASE_DIR, 'radius_neighbors_model.joblib'))
#bernoulli_nb_model = joblib.load(os.path.join(BASE_DIR, 'bernoulli_nb_model.joblib'))
print("Modèles chargés.")



# API Twilio 
import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

# API Twilio
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

twilio_phone_whatsapp = os.getenv(
    "TWILIO_PHONE_WHATSAPP",
    "whatsapp:+14155238886"
)

twilio_phone = os.getenv("TWILIO_PHONE")
police_number = os.getenv("POLICE_WHATSAPP_NUMBER")

client = Client(account_sid, auth_token)
client = Client(account_sid, auth_token)


# Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def upload_to_drive(file_path):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    permission = {'type': 'anyone', 'role': 'reader'}
    service.permissions().create(fileId=file_id, body=permission).execute()
    share_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    print(f"Fichier uploadé : {share_url}")
    return share_url

def upload_audio(file_path):
    try:
        return upload_to_drive(file_path)
    except Exception as e:
        print(f"Erreur upload audio Google Drive : {e}")
        return None

# Enregistrement audio avec préchauffage micr
def record_audio(duration=15, fs=22050, filename="recorded.wav", warmup=2):
    print(" Activation du micro...")
    try:
        # préchauffage micro
    #    sd.rec(int(warmup * fs), samplerate=fs, channels=1)
        sd.rec(int(warmup * fs), samplerate=fs, channels=1, device=1)
        sd.wait()
        print(" Micro prêt, enregistrement principal...")

       # recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=1)

        sd.wait()

        # vérification du signal
        if np.max(np.abs(recording)) < 0.01:
            print(" Aucun son détecté, nouvelle tentative...")
            sd.sleep(500)
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()

        sf.write(filename, recording, fs)
        if os.path.getsize(filename) < 10000:
            print(" Fichier audio suspectement vide.")
        else:
            print(f" Audio enregistré : {filename}")
        return filename

    except Exception as e:
        print(f"Erreur enregistrement audio : {e}")
        return None

#  Détection et analyse 
def extract_mfcc_features(file_path, n_mfcc=40, duration=2.0, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

def detect_scream(audio_features):
    # Liste des modèles et de leurs prédictions
    model_predictions = {}

    # Modèles Scikit-learn
    models = {
       # "SVM": svm_model,
     #   "LinearSVC": linear_svc_model,
        "RandomForest": rf_model,
        "ExtraTrees": et_model,
        "GradientBoosting": gb_model,
      #  "AdaBoost": ada_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
      #  "KNN": knn_model,
       # "DecisionTree": dt_model,
      #  "NaiveBayes": nb_model,
      #  "LogisticRegression": lr_model,
      #  "RidgeClassifier": ridge_model,
      #  "Perceptron": perc_model,
        "MLPClassifier": mlp_model,
      #  "LDA": lda_model,
      #  "QDA": qda_model,
      #  "SGDClassifier": sgd_model,
        "Bagging": bagging_model,
      #  "NuSVC": nusvc_model,
     #   "CalibratedSVC": calibrated_svc_model,
        
      
    }

    # Prédictions scikit-learn
    for name, model in models.items():
        try:
            pred = model.predict(audio_features)[0]
            model_predictions[name] = int(pred)
        except Exception as e:
            model_predictions[name] = -1  # -1 = erreur modèle

    # Modèles Keras
    try:
        mpn_pred = mpn_model.predict(audio_features)[0]
        model_predictions["MPN"] = int(np.argmax(mpn_pred))
    except:
        model_predictions["MPN"] = -1

    try:
        cnn_pred = cnn_model.predict(audio_features)[0]
        model_predictions["CNN"] = int(np.argmax(cnn_pred))
    except:
        model_predictions["CNN"] = -1

    # Regrouper les résultats
    scream_models = [name for name, pred in model_predictions.items() if pred == 1]
    no_scream_models = [name for name, pred in model_predictions.items() if pred == 0]
    failed_models = [name for name, pred in model_predictions.items() if pred == -1]

    # Comptage du nombre de détections
    scream_count = len(scream_models)

    # Niveau global de risque
    if scream_count >= 7:
        risk_level = "Haut risque"
    elif 5<= scream_count < 7:
        risk_level = "Risque moyen"
    else:
        risk_level = "Aucun risque"
                                                                                                                                                                                                                                                    
    #  Affichage des détails en console
    print("RÉSULTATS DÉTAILLÉS")
    print(f" Niveau global : {risk_level}")
    print(f" Modèles ayant détecté un cri ({len(scream_models)}): {', '.join(scream_models) or 'Aucun'}")
    print(f" Modèles sans détection ({len(no_scream_models)}): {', '.join(no_scream_models) or 'Aucun'}")
    if failed_models:
        print(f" Modèles en erreur ({len(failed_models)}): {', '.join(failed_models)}")
    print("\n")

    return risk_level

def get_location():
    g = geocoder.ip('me')
    if g.ok:
        latlng = g.latlng
        geolocator = Nominatim(user_agent="my_app", timeout=10)
        location = geolocator.reverse(f"{latlng[0]}, {latlng[1]}", language='fr')
        if location and location.raw and 'address' in location.raw:
            addr = location.raw['address']
            quartier = addr.get('suburb') or addr.get('neighbourhood') or addr.get('city_district')
            ville = addr.get('city') or addr.get('town') or addr.get('village')
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={latlng[0]},{latlng[1]}"
            if quartier:
                return f"{quartier}, {ville} - Voir sur Google Maps: {google_maps_url}"
            return f"{ville} - Voir sur Google Maps: {google_maps_url}"
    return "Localisation inconnue"

def send_alert(location, level, alert_time, audio_file):
    """
    Envoie :
      - un message WhatsApp avec l'audio
      - un SMS texte sans audio
    """
    results = []
    try:
        # Uploader l'audio pour WhatsApp
        audio_url = upload_audio(audio_file)
        if not audio_url:
            results.append("Impossible d'uploader l'audio pour WhatsApp")
        else:
            # Envoi WhatsApp
            try:
                message_wa = client.messages.create(
                    body=f"Alerte à {alert_time} : {level} de cri détecté à {location}. Audio : {audio_url}",
                    from_=twilio_phone_whatsapp,
                    to=police_number
                )
                results.append(f"Message WhatsApp envoyé : {message_wa.sid}")
            except Exception as e:
                results.append(f"Échec envoi WhatsApp : {str(e)}")

        # Envoi SMS texte (sans audio)
        try:
            message_sms = client.messages.create(
                body=f"Alerte à {alert_time} : {level} de cri détecté à {location}.",
                from_=twilio_phone,
                to="+22897839288"
            )
            results.append(f"SMS envoyé : {message_sms.sid}")
        except Exception as e:
            results.append(f"Échec envoi SMS : {str(e)}")

        return " | ".join(results)

    except Exception as e:
        return f"Erreur globale send_alert : {str(e)}"
# Enregistrement vidéo avec audio

def record_video_with_audio(duration=10, output_path="alert_video.mp4"):
    """
    Enregistre la vidéo depuis la webcam Logi C270 avec le son du microphone intégré.
    Nécessite ffmpeg installé sur le système.
    """

    # Vérifie que ffmpeg est disponible
    if not shutil.which("ffmpeg"):
        print(" ffmpeg n'est pas installé ou non détecté dans le PATH.")
        return None

    # Commande adaptée à ton matériel
    cmd = [
        "ffmpeg",
        "-y",  # écrase si le fichier existe
        "-f", "dshow",
        "-i", "video=Logi C270 HD WebCam:audio=Microphone (Logi C270 HD WebCam)",
        "-t", str(duration),
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-acodec", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        output_path
    ]

    print(" Enregistrement vidéo + audio en cours...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Vidéo enregistrée avec audio : {output_path}")

    return output_path



#  Fusion audio + vidéo 
def merge_audio_video(video_path,  output_path="final_video.mp4"):
    if not os.path.exists(video_path):
        print(" Vidéo ou audio manquant pour fusion")
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
     #   "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f" Vidéo finale avec audio : {output_path}")
    return output_path

# Modification de process_audio pour inclure vidéo 
def process_audio(audio_file):
    if audio_file is None:
        return "Aucun son détecté", "Localisation non déterminée", "Aucun envoi"

    time.sleep(0.5)
    features = extract_mfcc_features(audio_file)
    level = detect_scream(features)
    alert_time = datetime.datetime.now().strftime("%H:%M:%S")
    location = None
    wa_status = "Aucun envoi"

    if level != "Aucun risque":
        location = get_location()
        wa_status = send_alert(location, level, alert_time, audio_file)

        # 🔹 Si risque moyen ou haut, filmer 10s et envoyer
        if level in ["Risque moyen", "Haut risque"]:
            video_file = record_video_with_audio(duration=10)
            if video_file:
                final_video = merge_audio_video(video_file, output_path="final_video.mp4")
                if final_video:
                    try:
                        video_url = upload_to_drive(final_video)
                        client.messages.create(
                            body=f"🎥 Vidéo de la scène : {video_url}",
                            from_=twilio_phone_whatsapp,
                            to=police_number
                        )
                        print(" Lien vidéo envoyé à la police.")
                    except Exception as e:
                        print(f" Erreur envoi vidéo WhatsApp : {e}")

    return f"Niveau détecté : {level}", location or "Non déterminée", wa_status

# Gradio 
with gr.Blocks() as app:
    gr.Markdown("## Détection automatique de cris avec envoi WhatsApp")
    gr.Markdown("Enregistre via le micro et envoie un message WhatsApp avec l'audio si un cri est détecté.")
    
    audio_input = gr.Audio(type="filepath", sources=["microphone"], streaming=False)
    result = gr.Textbox(label="Résultat de l'analyse")
    location_out = gr.Textbox(label="Localisation détectée")
    wa_out = gr.Textbox(label="Statut WhatsApp")
    
    audio_input.change(fn=process_audio, inputs=audio_input, outputs=[result, location_out, wa_out])

if __name__ == "__main__":
    app.launch()
