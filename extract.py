# Ce code prend des fichiers audio (classés en positif/négatif), extrait pour chacun 40 MFCCs moyens, les associe à un label, et enregistre le tout dans un fichier CSV pour un futur apprentissage automatique
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm   # pour afficher une barre de progression lors des boucles

# Configuration
DATA_DIR = 'datae'  # Répertoire contenant les sous-dossiers (neutre, cri, etc.)
LABELS = {'datanegatif': 0, 'datapositif': 1}  # Associe chaque dossier à un label
N_MFCC = 40  # Nombre de coefficients MFCC à extraire
DURATION = 2.0  # Durée fixe des audios en secondes
SR = 22050  # Taux d'échantillonnage
# Liste pour stocker les données
data = []
print("Extraction des MFCCs...")
# Boucle de traitement des fichiers
for category in LABELS:
    label = LABELS[category]
    folder = os.path.join(DATA_DIR, category)

    for filename in tqdm(os.listdir(folder), desc=f"Traitement de {category}"):
        if filename.endswith(".wav"):  # on parcours les fichier .wav
            file_path = os.path.join(folder, filename)
            try:
                y, sr = librosa.load(file_path, sr=SR, duration=DURATION)  # charge l’audio en fixant le taux d’échantillonnage et la durée
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # extrait les MFCC (caractéristiques audio qui représentent le timbre du son)                
                # on prend la moyenne dans le temps de chaque coefficient (donc on obtient un vecteur de 40 valeurs)
                mfcc_mean = np.mean(mfcc, axis=1)
                #  on ajoute le label au vecteur MFCC.
                data.append(np.append(mfcc_mean, label))
            except Exception as e:
                print(f"Erreur avec {file_path} : {e}")
# Création du DataFrame
column_names = [f"mfcc_{i+1}" for i in range(N_MFCC)] + ['label']
df = pd.DataFrame(data, columns=column_names)  # On crée un DataFrame pandas avec des colonnes
# Sauvegarde
df.to_csv("mfcc_datavie.csv", index=False)
print(" Fichier mfcc_datae.csv généré avec succès.")
