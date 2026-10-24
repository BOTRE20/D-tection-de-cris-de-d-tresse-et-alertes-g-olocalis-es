import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from keras.models import Sequential
from keras.layers import Input, Dense

#  Chargement des données

df = pd.read_csv('mfcc_datavie.csv')
X = np.array(df.drop('label', axis=1))  # Variables explicatives (MFCCs)
y = np.array(df['label'])  # Labels (0 ou 1)
#y_categorical = to_categorical(y)
print(" Classes uniques :", np.unique(y))
print(" Nombre d’occurrences par classe :\n", pd.Series(y).value_counts())

#  Normalisation / Standardisation des MFCCs
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)  # Normalisation

# Séparation train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Répartition classes dans train avant SMOTE :", pd.Series(y_train).value_counts())

#Équilibrage avec SMOTE sur le train uniquement
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Conversion en one-hot encoding après SMOTE
y_categorical = to_categorical(y_train_bal)

print("Répartition classes dans train après SMOTE :", pd.Series(y_train_bal).value_counts())

#  Modèle MPN (réseau de neurones simple)
mpn_model = Sequential([
    Input(shape=(40,)),
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
mpn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mpn_model.fit(X_train_bal, y_categorical, epochs=30, batch_size=32, verbose=1)
mpn_model.save('mpn_model.h5')
print(" Modèle MPN sauvegardé dans 'mpn_model.h5'")

#  Modèle CNN (réseau de neurones convolutionnel)
X_cnn = np.expand_dims(X_train_bal, axis=2)
cnn_model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    Conv1D(16, 3, activation='relu'),
    Flatten(),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_cnn, y_categorical, epochs=30, batch_size=32, verbose=1)
cnn_model.save('cnn_model.h5')
print(" Modèle CNN sauvegardé dans 'cnn_model.h5'")

# Modèle SVM (Support Vector Machine)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_bal, y_train_bal)
joblib.dump(svm_model, 'svm_model.joblib')
print(" Modèle SVM sauvegardé dans 'svm_model.joblib'")

# Modèle LinearSVC
lsvc_model = LinearSVC()
lsvc_model.fit(X_train_bal, y_train_bal)
joblib.dump(lsvc_model, 'linear_svc_model.joblib')
print(" Modèle LinearSVC sauvegardé dans 'linear_svc_model.joblib'")

#  Modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_bal, y_train_bal)
joblib.dump(rf_model, 'random_forest_model.joblib')
print(" Modèle Random Forest sauvegardé dans 'random_forest_model.joblib'")

# Modèle Extra Trees
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train_bal, y_train_bal)
joblib.dump(et_model, 'extra_trees_model.joblib')
print(" Modèle Extra Trees sauvegardé dans 'extra_trees_model.joblib'")

#  Modèle Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_bal, y_train_bal)
joblib.dump(gb_model, 'gradient_boosting_model.joblib')
print(" Modèle Gradient Boosting sauvegardé dans 'gradient_boosting_model.joblib'")

# Modèle AdaBoost
ada_model = AdaBoostClassifier()
ada_model.fit(X_train_bal,y_train_bal)
joblib.dump(ada_model, 'adaboost_model.joblib')
print(" Modèle AdaBoost sauvegardé dans 'adaboost_model.joblib'")

# Modèle XGBoost
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train_bal, y_train_bal)
joblib.dump(xgb_model, 'xgboost_model.joblib')
print(" Modèle XGBoost sauvegardé dans 'xgboost_model.joblib'")

# Modèle LightGBM
lgb_model = LGBMClassifier()
lgb_model.fit(X_train_bal, y_train_bal)
joblib.dump(lgb_model, 'lightgbm_model.joblib')
print(" Modèle LightGBM sauvegardé dans 'lightgbm_model.joblib'")

# Modèle KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_bal, y_train_bal)
joblib.dump(knn_model, 'knn_model.joblib')
print(" Modèle KNN sauvegardé dans 'knn_model.joblib'")

# Modèle Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_bal, y_train_bal)
joblib.dump(dt_model, 'decision_tree_model.joblib')
print(" Modèle Decision Tree sauvegardé dans 'decision_tree_model.joblib'")


# 14. Modèle Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_bal, y_train_bal)
joblib.dump(nb_model, 'naive_bayes_model.joblib')
print(" Modèle Naive Bayes sauvegardé dans 'naive_bayes_model.joblib'")


# Modèle Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_bal, y_train_bal)
joblib.dump(lr_model, 'logistic_regression_model.joblib')
print(" Modèle Logistic Regression sauvegardé dans 'logistic_regression_model.joblib'")


# 16. Modèle Ridge Classifier
ridge_model = RidgeClassifier()
ridge_model.fit(X_train_bal,y_train_bal)
joblib.dump(ridge_model, 'ridge_classifier_model.joblib')
print(" Modèle Ridge Classifier sauvegardé dans 'ridge_classifier_model.joblib'")


# 17. Modèle Perceptron
perc_model = Perceptron(max_iter=1000)
perc_model.fit(X_train_bal, y_train_bal)
joblib.dump(perc_model, 'perceptron_model.joblib')
print(" Modèle Perceptron sauvegardé dans 'perceptron_model.joblib'")

# 18. Modèle MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
mlp_model.fit(X_train_bal, y_train_bal)
joblib.dump(mlp_model, 'mlp_classifier_model.joblib')
print(" Modèle MLPClassifier sauvegardé dans 'mlp_classifier_model.joblib'")

# 19. Modèle Linear Discriminant Analysis (LDA)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_bal, y_train_bal)
joblib.dump(lda_model, 'lda_model.joblib')
print(" Modèle LDA sauvegardé dans 'lda_model.joblib'")

# 20. Modèle Quadratic Discriminant Analysis (QDA)
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train_bal, y_train_bal)
joblib.dump(qda_model, 'qda_model.joblib')
print(" Modèle QDA sauvegardé dans 'qda_model.joblib'")

# 21. SGDClassifier
sgd_model = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train_bal, y_train_bal)
joblib.dump(sgd_model, 'sgd_classifier_model.joblib')
print(" Modèle SGDClassifier sauvegardé dans 'sgd_classifier_model.joblib'")

# 22. BaggingClassifier avec Decision Tree
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging_model.fit(X_train_bal, y_train_bal)
joblib.dump(bagging_model, 'bagging_model.joblib')
print(" Modèle BaggingClassifier sauvegardé dans 'bagging_model.joblib'")

# 24. NuSVC
nusvc_model = NuSVC(probability=True)
nusvc_model.fit(X_train_bal,y_train_bal)
joblib.dump(nusvc_model, 'nusvc_model.joblib')
print(" Modèle NuSVC sauvegardé dans 'nusvc_model.joblib'")

# 25. CalibratedClassifierCV (calibration d'un SVM)
calibrated_model = CalibratedClassifierCV(estimator=SVC(kernel='linear'), cv=5)
calibrated_model.fit(X_train_bal, y_train_bal)
joblib.dump(calibrated_model, 'calibrated_svc_model.joblib')
print(" Modèle CalibratedClassifierCV sauvegardé dans 'calibrated_svc_model.joblib'")

# 26. Radius Neighbors Classifier
radius_knn_model = RadiusNeighborsClassifier(radius=1.0)
radius_knn_model.fit(X_train_bal, y_train_bal)
joblib.dump(radius_knn_model, 'radius_neighbors_model.joblib')
print(" Modèle Radius Neighbors Classifier sauvegardé dans 'radius_neighbors_model.joblib'")

# 27. Bernoulli Naive Bayes
bnb_model = BernoulliNB()
bnb_model.fit(X_train_bal, y_train_bal)
joblib.dump(bnb_model, 'bernoulli_nb_model.joblib')
print(" Modèle Bernoulli Naive Bayes sauvegardé dans 'bernoulli_nb_model.joblib'")

# 28. Stochastic Gradient Descent Classifier
sgd_model = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train_bal, y_train_bal)
joblib.dump(sgd_model, 'sgd_classifier_model.joblib')
print(" Modèle SGD Classifier sauvegardé dans 'sgd_classifier_model.joblib'")

print("\n Entraînement terminé pour les 20 modèles de classification binaire !")

