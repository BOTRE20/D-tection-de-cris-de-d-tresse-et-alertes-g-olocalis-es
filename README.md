# ALERTE-IA

**Pipeline de classification audio, expérimentation multi-modèles et prototype d’alertes géolocalisées.**

ALERTE-IA explore la détection de cris à partir d’enregistrements audio, depuis la préparation des données jusqu’à l’intégration des prédictions dans des interfaces interactives. Le projet associe traitement du signal, apprentissage supervisé, réseaux de neurones et intégration de services de notification.

Deux variantes applicatives sont présentes : une interface **Streamlit** utilisant un CNN et une interface **Gradio** combinant les prédictions de neuf modèles, avec une capture vidéo locale déclenchée après détection.

> Statut : prototype expérimental. Les scripts d’extraction, d’entraînement et d’inférence sont disponibles ; les données, poids entraînés et rapports d’évaluation ne sont pas joints. Aucun niveau de performance ni fonctionnement de bout en bout n’est attesté ici. L’enregistrement est déclenché par l’utilisateur, sans surveillance audio continue.

## Réalisations techniques

- Construction d’un jeu de caractéristiques tabulaires à partir de fichiers audio : 40 MFCC moyens par fichier, exportés en CSV avec leur étiquette.
- Mise en place d’une séparation stratifiée entraînement/test de 80/20 et d’un rééquilibrage SMOTE appliqué uniquement au jeu d’entraînement.
- Implémentation de l’entraînement de **25 modèles distincts**, comprenant des méthodes classiques, des ensembles, un réseau dense et un CNN 1D.
- Sérialisation des modèles avec Joblib et Keras.
- Implémentation d’une décision par comptage des prédictions positives de **neuf modèles actifs** dans la variante Gradio.
- Intégration de la localisation, des API Twilio et Google Drive, ainsi que d’une capture vidéo avec FFmpeg.

## Pipeline de données et entraînement

### Extraction — `ba_extractm.py`

Le script parcourt les fichiers `.wav` des dossiers suivants :

| Dossier | Étiquette |
| --- | --- |
| `datae/datanegatif/` | `0` — classe négative |
| `datae/datapositif/` | `1` — classe positive, interprétée comme un cri |

Chaque fichier est chargé à **22 050 Hz**, sur une durée maximale de **2 secondes**. Les **40 MFCC** sont moyennés dans le temps, puis exportés dans `mfcc_datavie.csv` avec une colonne `label`. Les extraits courts ne sont pas explicitement complétés jusqu’à deux secondes. Le message final de console mentionne un autre nom de CSV ; le fichier effectivement écrit est bien `mfcc_datavie.csv`.

### Entraînement — `entrain_20.py`

Le script charge le CSV, réserve 20 % des observations pour le test avec `stratify=y` et `random_state=42`, puis applique SMOTE au jeu d’entraînement seulement. La standardisation apparaît dans le code, mais elle est commentée.

| Famille | Modèles implémentés |
| --- | --- |
| Réseaux de neurones | Réseau dense Keras nommé MPN, CNN 1D, MLPClassifier |
| Machines à vecteurs de support | SVC linéaire, LinearSVC, NuSVC, SVC calibré |
| Arbres et ensembles | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, Bagging |
| Voisinage | KNN, Radius Neighbors |
| Modèles linéaires | Logistic Regression, Ridge Classifier, Perceptron, SGDClassifier |
| Bayésiens | GaussianNB, BernoulliNB |
| Analyse discriminante | LDA, QDA |

Le réseau dense possède deux couches cachées de 64 et 32 neurones. Le CNN utilise deux couches Conv1D de 32 et 16 filtres, un aplatissement, un dropout de 0,3 et une sortie softmax à deux classes. Les deux réseaux Keras sont configurés pour **30 époques**, avec des lots de **32** observations et l’optimiseur Adam.

Malgré le nom du fichier et son message final, le script définit 25 modèles distincts. SGDClassifier est entraîné une seconde fois et son fichier est écrasé. Le CNN convolue le vecteur de coefficients MFCC moyens ; il ne traite pas un spectrogramme conservant son axe temporel.

### Évaluation : état actuel

Le jeu de test est créé, mais aucune évaluation sur ce jeu n’est exécutée dans le script. Les métriques d’entraînement Keras ne constituent pas une mesure des performances sur des données inédites. Le projet prépare une expérimentation multi-modèles, sans fournir encore de classement comparatif validé.

Pour compléter l’étude, mesurer sur le test réservé : précision, rappel de la classe cri, F1-score, matrice de confusion et temps d’inférence. Si plusieurs extraits proviennent d’une même source, séparer les données par source pour limiter les fuites entre entraînement et test.

## Variante Gradio — `test.py`

La variante Gradio reçoit un enregistrement du navigateur et extrait les mêmes caractéristiques MFCC. Sept modèles classiques sont actifs dans la décision : Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, MLPClassifier et Bagging. Le réseau dense Keras et le CNN portent le total à neuf.

| Prédictions positives | Libellé affiché |
| --- | --- |
| 7 à 9 | Haut risque |
| 5 à 6 | Risque moyen |
| 0 à 4 | Aucun risque |

Ces seuils sont des règles du prototype, sans validation documentée de leur lien avec un danger réel. Les modèles en erreur reçoivent la valeur `-1` et ne participent pas au compte positif ; les seuils ne sont pas ajustés au nombre de modèles disponibles.

En cas de risque moyen ou haut, le script tente d’envoyer l’alerte audio, puis de capturer **10 secondes de vidéo avec son** et d’en partager le lien par WhatsApp. La capture utilise FFmpeg, DirectShow sous Windows et une webcam Logi C270 nommée explicitement dans le code. Elle filme la scène après l’analyse, sans tampon vidéo antérieur à la détection. La fonction `merge_audio_video` recopie la piste vidéo et réencode l’audio déjà présent ; elle ne fusionne pas l’enregistrement initial avec la vidéo.

La localisation de cette variante est estimée à partir de l’**adresse IP de la machine exécutant Python**, puis enrichie avec Nominatim. Elle ne correspond pas à une mesure GPS précise du navigateur. Le microphone de l’interface est celui du navigateur ; la webcam sollicitée par FFmpeg est celle de la machine exécutant Python.

## Variante Streamlit — `app(1).py`

## Fonctionnalités

- Enregistrement audio depuis le navigateur et réécoute dans l’interface.
- Extraction de caractéristiques MFCC avec Librosa.
- Classification audio avec un modèle chargé par TensorFlow/Keras.
- Récupération des coordonnées du navigateur, sous réserve d’autorisation.
- Géocodage inverse avec Nominatim pour identifier le quartier et la ville.
- Construction d’un lien Google Maps et affichage de la précision GPS.
- Envoi de notifications SMS et WhatsApp via Twilio.
- Téléversement de l’audio sur Google Drive et partage de son lien dans le message WhatsApp.
- Affichage des résultats et des statuts d’envoi dans Streamlit.

## Technologies

| Composant | Technologie |
| --- | --- |
| Interface | Streamlit |
| Calcul numérique | NumPy |
| Traitement audio | Librosa |
| Inférence | TensorFlow / Keras |
| Notifications | API Twilio |
| Localisation du navigateur | streamlit-js-eval |
| Géocodage inverse | Geopy / Nominatim |
| Stockage audio | API Google Drive |
| Authentification Google | OAuth 2.0 |

## Fonctionnement

1. L’utilisateur enregistre un extrait audio dans le navigateur.
2. L’application sauvegarde l’enregistrement dans un fichier WAV temporaire.
3. Librosa charge au maximum les deux premières secondes à 22 050 Hz.
4. Le programme extrait 40 coefficients MFCC et calcule leur moyenne temporelle, produisant un tableau de forme `(1, 40)`.
5. Le modèle prédit une classe. Le code considère l’indice `1` comme un cri et affiche « Haut risque » ; les autres indices donnent « Aucun risque ».
6. En cas de détection, l’application tente de récupérer la localisation, de téléverser l’audio puis d’envoyer les notifications.

Le message WhatsApp dépend de la réussite du téléversement sur Google Drive. L’envoi du SMS est tenté séparément, même si ce téléversement échoue. L’enregistrement partagé correspond au fichier complet, tandis que l’analyse porte uniquement sur les deux premières secondes.

## Fichiers nécessaires

| Fichier | Rôle | Disponibilité |
| --- | --- | --- |
| `app(1).py` | Application principale | Code fourni |
| `cnn_model.keras` | Modèle entraîné | À ajouter à côté du script |
| `credentials.json` | Identifiants OAuth Google | À créer localement, confidentiel |
| `token.json` | Jetons OAuth | Créé lors de l’authentification, confidentiel |
| `README.md` | Documentation | Présent document |

Les scripts `ba_extractm.py`, `entrain_20.py` et `testz.py` complètent cette application. Les données, modèles sauvegardés et métriques d’évaluation ne sont pas fournis.

## Installation locale — Streamlit

### 1. Préparer l’environnement

Installer Python dans une version compatible avec TensorFlow et l’environnement ayant servi à enregistrer le modèle. Les versions exactes des dépendances du projet ne sont pas précisées dans le code fourni.

Depuis le dossier du projet :

```bash
python -m venv .venv
```

Activer l’environnement sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

Ou sous Linux / macOS :

```bash
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install streamlit numpy librosa tensorflow twilio geopy streamlit-js-eval google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
```

Cette liste est déduite des imports ; elle ne constitue pas un environnement de versions validé. Utiliser une version de Streamlit proposant `st.audio_input`.

### 3. Ajouter le modèle

Placer `cnn_model.keras` dans le même dossier que `app(1).py`. Le modèle doit accepter les caractéristiques produites par `extract_mfcc_features` et utiliser une correspondance de classes compatible avec le code : l’indice `1` doit représenter un cri. Vérifier également que le prétraitement reproduit celui utilisé à l’entraînement.

### 4. Configurer Twilio

Le script fourni contient des identifiants et des numéros inscrits directement dans le code. **Renouveler le jeton exposé et supprimer les secrets du code avant toute publication.**

Configurer un compte Twilio, un expéditeur SMS et un expéditeur WhatsApp autorisé pour l’environnement de test. Utiliser uniquement des destinataires de test consentants. Le nom de variable `police_number` ne signifie pas qu’une intégration avec un service de police existe.

Pour externaliser les paramètres, remplacer le bloc de configuration Twilio par une lecture des variables d’environnement :

```python
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
twilio_phone_whatsapp = os.environ["TWILIO_WHATSAPP_FROM"]
twilio_phone = os.environ["TWILIO_SMS_FROM"]
police_number = os.environ["ALERT_WHATSAPP_TO"]
sms_recipient = os.environ["ALERT_SMS_TO"]

client = Client(account_sid, auth_token)
```

Dans `send_alert`, remplacer aussi le numéro SMS codé en dur par `to=sms_recipient`. Les numéros WhatsApp doivent inclure le préfixe `whatsapp:` ; les numéros SMS doivent utiliser le format international.

**Cette adaptation est à effectuer dans le script : le code actuel ne lit pas ces variables et ne charge pas automatiquement un fichier `.env`.**

### 5. Configurer Google Drive

- Préparer un projet Google avec l’API Google Drive activée et un client OAuth adapté à une application locale.
- Enregistrer les identifiants du client dans `credentials.json`.
- Placer ce fichier dans le dossier depuis lequel l’application sera lancée : le code utilise un chemin relatif pour les fichiers OAuth.
- Lors du premier téléversement, suivre l’autorisation Google dans le navigateur. Le code démarre le serveur OAuth local sur le port `8080`.
- Le fichier `token.json` est ensuite créé pour réutiliser l’autorisation.

Le périmètre demandé par le script est `https://www.googleapis.com/auth/drive.file`.

> Le code applique actuellement une permission `anyone / reader` à chaque audio téléversé. Toute personne disposant du lien peut donc y accéder. Réserver la démonstration à des enregistrements non sensibles et revoir ce partage avant un usage réel.

### 6. Lancer l’application

Depuis le dossier contenant le script et les fichiers de configuration :

```bash
python -m streamlit run "app(1).py"
```

Ouvrir l’adresse affichée par Streamlit. Autoriser le microphone et, lorsque demandée, la géolocalisation. Pour un hébergement distant, prévoir HTTPS et adapter le flux OAuth local ; le script est conçu ici pour une démonstration locale.

## Utilisation

1. Enregistrer un extrait avec le composant audio.
2. Attendre la fin de l’analyse.
3. Consulter le niveau affiché, la localisation disponible et les statuts d’envoi.
4. Vérifier la réception sur les numéros de test configurés.

La création d’un message par l’API Twilio ne garantit pas sa livraison finale. Vérifier également son statut côté Twilio et sa réception effective.

## Limites identifiées

- Aucune précision, aucun rappel ni taux de faux positifs ne peut être établi à partir du seul script fourni.
- La classification est fondée sur `argmax`, sans seuil de confiance explicite.
- Une erreur d’inférence est actuellement convertie en « Aucun risque » : ce comportement doit être corrigé pour distinguer une panne d’une absence de détection.
- La géolocalisation peut être indisponible, refusée ou imprécise. Une alerte peut alors partir sans position exploitable.
- Le code ne comporte pas de mécanisme explicite de déduplication, de limitation de fréquence ou de reprise des alertes.
- Les fichiers WAV temporaires ne sont pas supprimés explicitement après traitement.
- Aucun mécanisme d’authentification utilisateur ni politique de conservation des audios n’est implémenté dans le script.
- Le fonctionnement complet nécessite le modèle, les autorisations du navigateur, une connexion Internet et la configuration des services externes.

## Vérifications avant démonstration

- Vérifier le chargement du modèle et la correspondance de ses classes.
- Tester un enregistrement contenant un cri et un enregistrement sans cri.
- Tester le refus de géolocalisation et les erreurs des services externes.
- Confirmer la réception des notifications sur des numéros de test.
- Vérifier l’accès au lien audio et supprimer les enregistrements de démonstration après usage.

Ces vérifications sont proposées ; aucune exécution du modèle ni aucun test d’envoi n’est attesté par cette documentation.

## Préparer le dépôt GitHub

Ajouter au minimum les exclusions suivantes à un fichier `.gitignore` :

```gitignore
.venv/
__pycache__/
*.pyc
.env
.env.*
!.env.example
credentials.json
token.json
.streamlit/secrets.toml
*.wav
```

Un `.gitignore` ne retire pas des secrets déjà présents dans l’historique Git : renouveler les secrets exposés et nettoyer l’historique concerné avant publication. Ne pas publier de numéros personnels ni d’enregistrements sensibles.

## Exécuter le pipeline et la variante Gradio

Les commandes suivantes décrivent l’ordre prévu. **Les écarts d’intégration listés ci-dessous doivent être corrigés avant un essai de bout en bout.**

En complément des dépendances Streamlit :

```bash
python -m pip install pandas tqdm scikit-learn imbalanced-learn xgboost lightgbm joblib keras gradio opencv-python sounddevice soundfile geocoder
```

Après constitution des dossiers de données :

```bash
python ba_extractm.py
python entrain_20.py
```

Après harmonisation des chemins et formats des modèles, configuration des services et installation de FFmpeg dans le `PATH` :

```bash
python testz.py
```

La capture vidéo nécessite Windows et l’adaptation du nom de périphérique FFmpeg au matériel utilisé. La fonction auxiliaire `record_audio` impose également `device=1`, mais elle n’est pas appelée par le parcours Gradio actuel. Le script charge dès son démarrage plusieurs modèles non utilisés par le vote : leurs fichiers restent nécessaires tant que ces chargements ne sont pas retirés.

### Écarts à harmoniser

| Point | État observé | Action nécessaire |
| --- | --- | --- |
| Format Keras | L’entraînement sauvegarde `.h5`, les interfaces chargent `.keras` | Harmoniser les appels de sauvegarde et chargement ; ne pas simplement changer l’extension |
| Noms Joblib | Exemples : `random_forest_model.joblib` produit, `randomforest_model.joblib` attendu ; `linear_svc_model.joblib` produit, `linearsvc_model.joblib` attendu | Aligner tous les noms utilisés dans `testz.py` sur ceux sauvegardés |
| Entrée CNN | Entraînement en `(N, 40, 1)`, extraction d’inférence en `(1, 40)` | Ajouter explicitement l’axe canal avant la prédiction CNN et vérifier la compatibilité |
| Évaluation | `X_test` et `y_test` réservés mais non exploités | Ajouter l’évaluation indépendante et enregistrer les résultats |
| Erreurs d’inférence | Les erreurs peuvent réduire artificiellement le niveau affiché | Distinguer indisponibilité technique et résultat négatif |
| Secrets | Identifiants Twilio présents dans les deux interfaces, y compris des commentaires de `testz.py` | Renouveler les jetons concernés et externaliser la configuration |

Les extraits de configuration Streamlit proposés plus haut doivent aussi être adaptés à `testz.py`. Les scripts source n’ont pas été modifiés dans le cadre de cette documentation.

## Compétences mobilisées

**Traitement du signal audio**, préparation de données, extraction de caractéristiques, classification supervisée, gestion du déséquilibre de classes, réseaux de neurones, sérialisation de modèles, interfaces Python, intégration d’API et capture multimédia.

## Évolutions envisagées

- Séparer l’interface, l’inférence et les services de notification.
- Ajouter une gestion explicite des erreurs d’analyse et des tests automatisés.
- Évaluer le modèle sur un jeu de test documenté.
- Ajouter un seuil de confiance et une protection contre les alertes répétées.
- Sécuriser l’accès aux audios et définir leur durée de conservation.
- Nettoyer systématiquement les fichiers temporaires.

## Licence

Aucune licence n’est indiquée dans le fichier source fourni. Ajouter un fichier `LICENSE` avant de définir les conditions de réutilisation publique du projet.
