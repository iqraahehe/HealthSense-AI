import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    print("Cleaning data...")
    df = df.copy()
    
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} duplicate rows")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing} missing values -> filled with median")
            df[col] = df[col].fillna(df[col].median())
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('Female')
    
    if 'glucose' in df.columns:
        df = df[(df['glucose'] >= 20) & (df['glucose'] <= 600)]
    
    if 'blood_pressure' in df.columns:
        df = df[(df['blood_pressure'] >= 40) & (df['blood_pressure'] <= 250)]
    
    if 'heart_rate' in df.columns:
        df = df[(df['heart_rate'] >= 30) & (df['heart_rate'] <= 200)]
    
    if 'hemoglobin' in df.columns:
        df = df[(df['hemoglobin'] >= 3) & (df['hemoglobin'] <= 25)]
    
    print(f"Data cleaned. Final size: {len(df)} rows")
    
    return df

def engineer_features(df):
    print("Engineering features...")
    
    df = df.copy()
    
    if 'glucose' in df.columns:
        df['glucose_category'] = pd.cut(
            df['glucose'], 
            bins=[0, 100, 125, 1000],
            labels=['normal', 'prediabetes', 'diabetes']
        )
    
    if 'blood_pressure' in df.columns:
        df['bp_category'] = pd.cut(
            df['blood_pressure'],
            bins=[0, 120, 140, 1000],
            labels=['normal', 'elevated', 'high']
        )
    
    print(f"Feature engineering complete")
    
    return df

def prepare_for_training(df, feature_columns, target_column):
    print("\nPreparing data for training...")
    
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'Male': 1.0, 'Female': 0.0, 'male': 1.0, 'female': 0.0}).fillna(0.0)
    
    print(f"  Features: {feature_columns}")
    print(f"  Target: {target_column}")
    print(f"  Samples: {len(X)}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n  Label encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"    {label} -> {i}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n  Feature scaling complete")
    print(f"    Mean: {X_scaled.mean(axis=0).round(3)}")
    print(f"    Std:  {X_scaled.std(axis=0).round(3)}")
    
    unique, counts = np.unique(y_encoded, return_counts=True)
    print(f"\n  Class distribution:")
    for label, count in zip(label_encoder.classes_, counts):
        print(f"    {label}: {count} ({count/len(y)*100:.1f}%)")
    
    return X_scaled, y_encoded, scaler, label_encoder

def load_and_prepare(filepath, feature_columns, target_column):
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    df = clean_data(df)
    
    X, y, scaler, label_encoder = prepare_for_training(
        df, feature_columns, target_column
    )
    
    return X, y, scaler, label_encoder, df

def save_preprocessed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

def get_feature_statistics(df, feature_columns, target_column):
    print("\n" + "="*70)
    print("FEATURE STATISTICS BY CLASS")
    print("="*70)
    
    for disease in df[target_column].unique():
        print(f"\n{disease.upper()}:")
        subset = df[df[target_column] == disease][feature_columns]
        numeric_subset = subset.select_dtypes(include=[np.number])
        print(numeric_subset.describe().round(2))
    
    print("="*70 + "\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data preparation')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data CSV file')
    
    args = parser.parse_args()
    
    feature_columns = ['gender', 'glucose', 'blood_pressure', 'heart_rate', 
                      'hemoglobin', 'cholesterol', 'bmi', 'age']
    target_column = 'disease'
    
    X, y, scaler, label_encoder, df = load_and_prepare(
        args.data, feature_columns, target_column
    )
    
    get_feature_statistics(df, feature_columns, target_column)
    
    print(f"\nData preparation test complete!")
    print(f"Final shape: X={X.shape}, y={y.shape}")