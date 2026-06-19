import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_diabetes_case():
    return {
        'gender': np.random.choice(['Male', 'Female']),
        'glucose': np.random.normal(180, 30),          
        'blood_pressure': np.random.normal(140, 15),   
        'heart_rate': np.random.normal(80, 10),
        'hemoglobin': np.random.normal(14, 1.5),       
        'cholesterol': np.random.normal(220, 25),      
        'bmi': np.random.normal(30, 5),                
        'age': np.random.normal(55, 12),
        'disease': 'diabetes'
    }

def generate_heart_case():
    return {
        'gender': np.random.choice(['Male', 'Female']),
        'glucose': np.random.normal(110, 20),
        'blood_pressure': np.random.normal(150, 20),   
        'heart_rate': np.random.normal(95, 12),        
        'hemoglobin': np.random.normal(14, 1.5),
        'cholesterol': np.random.normal(240, 30),      
        'bmi': np.random.normal(28, 4),                
        'age': np.random.normal(60, 10),
        'disease': 'heart'
    }

def generate_anemia_case():
    return {
        'gender': np.random.choice(['Male', 'Female']),
        'glucose': np.random.normal(95, 15),
        'blood_pressure': np.random.normal(105, 12),  
        'heart_rate': np.random.normal(85, 12),       
        'hemoglobin': np.random.normal(9, 1.5),       
        'cholesterol': np.random.normal(170, 20),
        'bmi': np.random.normal(22, 3),
        'age': np.random.normal(45, 15),
        'disease': 'anemia'
    }

def generate_healthy_case():
    return {
        'gender': np.random.choice(['Male', 'Female']),
        'glucose': np.random.normal(95, 10),           
        'blood_pressure': np.random.normal(115, 8),    
        'heart_rate': np.random.normal(75, 8),         
        'hemoglobin': np.random.normal(14.5, 1),       
        'cholesterol': np.random.normal(170, 15),      
        'bmi': np.random.normal(22, 2),                
        'age': np.random.normal(40, 15),
        'disease': 'healthy'
    }

def clip_values(data):
    data['glucose'] = np.clip(data['glucose'], 50, 400)
    data['blood_pressure'] = np.clip(data['blood_pressure'], 70, 200)
    data['heart_rate'] = np.clip(data['heart_rate'], 40, 150)
    data['hemoglobin'] = np.clip(data['hemoglobin'], 5, 20)
    data['cholesterol'] = np.clip(data['cholesterol'], 100, 350)
    data['bmi'] = np.clip(data['bmi'], 15, 50)
    data['age'] = np.clip(data['age'], 18, 90)
    return data

def generate_dataset(n_samples=500, diabetes_ratio=0.25, heart_ratio=0.25, 
                    anemia_ratio=0.20, healthy_ratio=0.30):
    
    n_diabetes = int(n_samples * diabetes_ratio)
    n_heart = int(n_samples * heart_ratio)
    n_anemia = int(n_samples * anemia_ratio)
    n_healthy = n_samples - n_diabetes - n_heart - n_anemia
    
    print(f"Generating {n_samples} samples:")
    print(f"  Diabetes: {n_diabetes}")
    print(f"  Heart: {n_heart}")
    print(f"  Anemia: {n_anemia}")
    print(f"  Healthy: {n_healthy}")
    
    data = []
    for _ in range(n_diabetes):
        data.append(clip_values(generate_diabetes_case()))
    
    for _ in range(n_heart):
        data.append(clip_values(generate_heart_case()))
    
    for _ in range(n_anemia):
        data.append(clip_values(generate_anemia_case()))
    
    for _ in range(n_healthy):
        data.append(clip_values(generate_healthy_case()))
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['glucose'] = df['glucose'].round(1)
    df['blood_pressure'] = df['blood_pressure'].round(0)
    df['heart_rate'] = df['heart_rate'].round(0)
    df['hemoglobin'] = df['hemoglobin'].round(1)
    df['cholesterol'] = df['cholesterol'].round(0)
    df['bmi'] = df['bmi'].round(1)
    df['age'] = df['age'].round(0)
    
    print(f"\nDataset generated successfully!")
    print(f"\nClass distribution:")
    print(df['disease'].value_counts())
    
    print(f"\nSample statistics:")
    numeric_df = df.drop(columns=['gender'])
    print(numeric_df.groupby(df['disease']).mean().round(2))
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic medical data for training'
    )
    parser.add_argument('--out', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--n', type=int, default=500,
                       help='Number of samples to generate (default: 500)')
    parser.add_argument('--diabetes', type=float, default=0.25,
                       help='Proportion of diabetes cases (default: 0.25)')
    parser.add_argument('--heart', type=float, default=0.25,
                       help='Proportion of heart disease cases (default: 0.25)')
    parser.add_argument('--anemia', type=float, default=0.20,
                       help='Proportion of anemia cases (default: 0.20)')
    
    args = parser.parse_args()
    
    healthy_ratio = 1.0 - args.diabetes - args.heart - args.anemia
    
    if healthy_ratio < 0:
        raise ValueError("Disease ratios sum to more than 1.0!")
    
    df = generate_dataset(
        n_samples=args.n,
        diabetes_ratio=args.diabetes,
        heart_ratio=args.heart,
        anemia_ratio=args.anemia,
        healthy_ratio=healthy_ratio
    )
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    print("\nFirst 5 samples:")
    print(df.head())

if __name__ == '__main__':
    main()