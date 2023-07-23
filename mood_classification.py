import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import librosa
import pandas as pd
import os
from subprocess import call


# Function to load audio file
def load_audio_file(file_path):
    data, _ = librosa.load(file_path, sr=None)
    return data


def extract_features_opensmile(wav_path, output_file, opensmile_path, config_file):
    conf_path = os.path.join(opensmile_path, 'config', config_file)
    command = [
        os.path.join(opensmile_path, 'bin', 'Win32', 'SMILExtract_Release.exe'),
        '-C', conf_path,
        '-I', wav_path,
        '-csvoutput', output_file,
        '-timestampcsv', '0'
    ]
    call(' '.join(command))

def extract_audio_features(filename):
    # Path to openSMILE
    opensmile_path = r"D:\Users\fernandez.laura\Downloads\opensmile-2.3.0"

    # Configuration file for eGeMAPS features
    config_file = r"gemaps\eGeMAPSv01a.conf"

    # Output file for features
    output_file = filename + '.csv'

    # Extract eGeMAPS features
    extract_features_opensmile(filename, output_file, opensmile_path, config_file)

    # Load eGeMAPS features from CSV file
    eGeMAPS_features = pd.read_csv(output_file).values

    # Extract MFCC features
    y, sr = librosa.load(filename)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr)

    # Combine eGeMAPS and MFCC features into one array
    combined_features = np.concatenate([eGeMAPS_features, mfcc_features], axis=1)

    return combined_features


# Prepare models
models = []
models.append(('SVC', SVC(probability=True)))
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))

# Define hyperparameters for GridSearchCV
hyperparameters = {
    'SVC': {'kernel': ['linear', 'rbf'], 'C': [1, 10]},
    'RF': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

# Encoding labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Best model and its score
best_model = None
best_score = 0
best_params = None

# Nested CV for model selection and hyperparameter tuning
inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=42)

for name, model in models:
    clf = GridSearchCV(model, hyperparameters[name], cv=inner_cv, scoring='accuracy')
    clf.fit(X, y_encoded)
    nested_score = cross_val_score(clf, X=X, y=y_encoded, cv=outer_cv).mean()
    print(f'Model {name} | Nested CV score: {nested_score}')
    if nested_score > best_score:
        best_score = nested_score
        best_model = clf
        best_params = clf.best_params_

print(f'Best Model: {best_model} | Best Score: {best_score} | Best Params: {best_params}')

# Save the best model for later use
joblib.dump(best_model, 'voice_mood_model.pkl')

# Function to predict mood from an audio file
def predict_mood(file_path):
    # Load the audio file
    data = load_audio_file(file_path)
    # Extract features
    features = extract_feature(data)
    # Load the saved model
    model = joblib.load('voice_mood_model.pkl')
    # Predict
    result = model.predict([features])
    # Also predict the probability
    result_proba = model.predict_proba([features])
    # Convert numerical labels back to original labels
    label_mapping = {label: index for index, label in enumerate(encoder.classes_)}
    return label_mapping[result[0]], result_proba
