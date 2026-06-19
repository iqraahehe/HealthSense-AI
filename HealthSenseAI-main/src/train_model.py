import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import argparse
from pathlib import Path
import json

FEATURE_COLUMNS = ['gender', 'glucose', 'blood_pressure', 'heart_rate', 'hemoglobin', 
                   'cholesterol', 'bmi', 'age']
TARGET_COLUMN = 'disease'

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    print("\nPreprocessing data...")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'Male': 1.0, 'Female': 0.0, 'male': 1.0, 'female': 0.0}).fillna(0.0)
    
    X = X.fillna(X.median())
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, scaler, label_encoder

def train_models(X_train, y_train):
    print("\nTraining AI Models...")
    
    print("Training Gaussian Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return nb_model, rf_model

def save_artifacts(nb_model, rf_model, scaler, label_encoder, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump(nb_model, output_dir / 'nb_model.pkl')
    joblib.dump(rf_model, output_dir / 'rf_model.pkl')
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    joblib.dump(label_encoder, output_dir / 'label_encoder.pkl')
    print(f"\nAll models and artifacts saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train disease prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--out', type=str, default='models', help='Output directory')
    args = parser.parse_args()
    
    df = load_data(args.data)
    X, y, scaler, label_encoder = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    nb_model, rf_model = train_models(X_train, y_train)
    
    print("\n--- Naive Bayes Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, nb_model.predict(X_test)):.4f}")
    
    print("\n--- Random Forest Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")
    
    save_artifacts(nb_model, rf_model, scaler, label_encoder, args.out)

if __name__ == '__main__':
    main()