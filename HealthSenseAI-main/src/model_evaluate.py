import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_artifacts(model_dir):
    model_dir = Path(model_dir)
    
    model = joblib.load(model_dir / 'rf_model.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
    
    print("Model artifacts loaded successfully")
    return model, scaler, label_encoder

def evaluate_on_data(model, scaler, label_encoder, data_path):
    print(f"\nEvaluating model on {data_path}...")
    
    df = pd.read_csv(data_path)
    
    feature_columns = ['gender', 'glucose', 'blood_pressure', 'heart_rate', 
                      'hemoglobin', 'cholesterol', 'bmi', 'age']
    
    X = df[feature_columns].copy()
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'Male': 1.0, 'Female': 0.0, 'male': 1.0, 'female': 0.0}).fillna(0.0)
    
    X_values = X.values
    y_true_labels = df['disease'].values
    
    X_scaled = scaler.transform(X_values)
    
    y_pred_encoded = model.predict(X_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_labels, y_pred_labels, 
                         labels=label_encoder.classes_)
    print(cm)
    
    plot_confusion_matrix(cm, label_encoder.classes_)
    
    return accuracy, y_pred_labels

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = Path('reports') / 'confusion_matrix.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {output_path}")
    plt.close()

def predict_single_case(model, scaler, label_encoder, case_data):
    feature_columns = ['gender', 'glucose', 'blood_pressure', 'heart_rate', 
                      'hemoglobin', 'cholesterol', 'bmi', 'age']
    
    features = []
    for col in feature_columns:
        features.append(case_data.get(col, 0))
    
    features = np.array(features).reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    
    pred_encoded = model.predict(features_scaled)[0]
    pred_proba = model.predict_proba(features_scaled)[0]
    
    prediction = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = max(pred_proba)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print("\nInput Data:")
    for key, value in case_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction: {prediction.upper()}")
    print(f"Confidence: {confidence*100:.1f}%")
    
    print("\nClass Probabilities:")
    for label, prob in zip(label_encoder.classes_, pred_proba):
        print(f"  {label}: {prob*100:.1f}%")
    
    print("="*60)
    
    return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing model artifacts')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to test data CSV')
    parser.add_argument('--predict', action='store_true',
                       help='Make a single prediction with custom data')
    
    args = parser.parse_args()
    model, scaler, label_encoder = load_model_artifacts(args.model_dir)
    
    if args.data:
        evaluate_on_data(model, scaler, label_encoder, args.data)
    
    elif args.predict:
        print("\nEnter patient data:")
        gender_input = input("Gender (Male/Female): ").strip()
        gender_val = 1.0 if gender_input.lower() == 'male' else 0.0
        
        case_data = {
            'gender': gender_val,
            'glucose': float(input("Glucose (mg/dL): ")),
            'blood_pressure': float(input("Blood Pressure (mmHg): ")),
            'heart_rate': float(input("Heart Rate (BPM): ")),
            'hemoglobin': float(input("Hemoglobin (g/dL): ")),
            'cholesterol': float(input("Cholesterol (mg/dL): ")),
            'bmi': float(input("BMI: ")),
            'age': float(input("Age (years): "))
        }
        
        predict_single_case(model, scaler, label_encoder, case_data)
    
    else:
        print("\nPlease specify --data for evaluation or --predict for single prediction")

if __name__ == '__main__':
    main()