import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

HEALTHY_RANGES = {
    'glucose': (70, 120),         
    'blood_pressure': (90, 130),  
    'heart_rate': (60, 100),      
    'hemoglobin': (12.0, 17.0),   
    'cholesterol': (125, 200),    
    'bmi': (18.5, 24.9),          
    'age': (0, 120)               
}


def load_report(filepath):
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        if len(df) == 0:
            raise ValueError("CSV file is empty")
        return df.iloc[0].to_dict()
    
    elif filepath.suffix == '.txt':
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    try:
                        data[key] = float(value)
                    except ValueError:
                        continue
        return data
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def validate_data(data):

    errors = []
    
    required_fields = ['glucose', 'blood_pressure', 'heart_rate', 
                      'hemoglobin', 'cholesterol', 'bmi']
    
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    for key, value in data.items():
        if key in HEALTHY_RANGES:
            min_val, max_val = HEALTHY_RANGES[key]
            if key == 'glucose' and (value < 20 or value > 600):
                errors.append(f"{key} value {value} is not physiologically possible")
            elif key == 'blood_pressure' and (value < 40 or value > 250):
                errors.append(f"{key} value {value} is not physiologically possible")
            elif key == 'heart_rate' and (value < 30 or value > 200):
                errors.append(f"{key} value {value} is not physiologically possible")
            elif key == 'hemoglobin' and (value < 3 or value > 25):
                errors.append(f"{key} value {value} is not physiologically possible")
    
    return len(errors) == 0, errors


def check_healthy(data):

    violations = []
    
    for key, value in data.items():
        if key in HEALTHY_RANGES:
            min_val, max_val = HEALTHY_RANGES[key]
            if value < min_val:
                violations.append(f"{key}: {value:.1f} (LOW, normal: {min_val}-{max_val})")
            elif value > max_val:
                violations.append(f"{key}: {value:.1f} (HIGH, normal: {min_val}-{max_val})")
    
    is_healthy = len(violations) == 0
    
    return is_healthy, violations


def prepare_features(data, feature_columns):

    features = []
    for col in feature_columns:
        features.append(data.get(col, 0))
    
    return np.array(features).reshape(1, -1)


def visualize_report(data, save_path=None):

    indicators = {k: v for k, v in data.items() if k in HEALTHY_RANGES and k != 'age'}
    
    if not indicators:
        print("No health indicators to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = []
    values = []
    colors = []
    
    for key, value in indicators.items():
        names.append(key.replace('_', ' ').title())
        values.append(value)
        
        if key in HEALTHY_RANGES:
            min_val, max_val = HEALTHY_RANGES[key]
            if min_val <= value <= max_val:
                colors.append('green')
            elif value < min_val:
                colors.append('blue')
            else:
                colors.append('red')
        else:
            colors.append('gray')
    
    bars = ax.barh(names, values, color=colors, alpha=0.7)
    
    for i, (key, value) in enumerate(indicators.items()):
        if key in HEALTHY_RANGES:
            min_val, max_val = HEALTHY_RANGES[key]
            ax.axvline(min_val, color='green', linestyle='--', alpha=0.3)
            ax.axvline(max_val, color='green', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Value')
    ax.set_title('Medical Report Analysis')
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Normal'),
        Patch(facecolor='blue', alpha=0.7, label='Low'),
        Patch(facecolor='red', alpha=0.7, label='High')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_report_summary(data, prediction=None, confidence=None):
    """Print a formatted summary of the medical report."""
    print("\n" + "="*60)
    print("MEDICAL REPORT SUMMARY")
    print("="*60)
    
    for key, value in data.items():
        if key in HEALTHY_RANGES and key != 'age':
            min_val, max_val = HEALTHY_RANGES[key]
            status = "✓" if min_val <= value <= max_val else "✗"
            print(f"{status} {key.replace('_', ' ').title():20s}: {value:6.1f}  (normal: {min_val}-{max_val})")
    
    if 'age' in data:
        print(f"  {'Age':20s}: {data['age']:.0f} years")
    
    if prediction:
        print(f"\n{'PREDICTION':20s}: {prediction.upper()}")
        if confidence:
            print(f"{'CONFIDENCE':20s}: {confidence}")
    
    print("="*60 + "\n")


def get_disease_info(disease):
    """Get information about a predicted disease."""
    disease_info = {
        'diabetes': {
            'name': 'Diabetes Mellitus',
            'description': 'A metabolic disorder characterized by high blood glucose levels.',
            'key_indicators': ['High glucose', 'Often high cholesterol', 'Often high BMI'],
            'recommendations': [
                'Consult an endocrinologist',
                'Monitor blood glucose regularly',
                'Follow a diabetes-friendly diet',
                'Maintain regular physical activity'
            ]
        },
        'heart': {
            'name': 'Cardiovascular Disease',
            'description': 'Conditions affecting the heart and blood vessels.',
            'key_indicators': ['High blood pressure', 'Elevated heart rate', 'High cholesterol'],
            'recommendations': [
                'Consult a cardiologist',
                'Monitor blood pressure regularly',
                'Follow a heart-healthy diet',
                'Reduce sodium intake',
                'Manage stress levels'
            ]
        },
        'anemia': {
            'name': 'Anemia',
            'description': 'A condition where blood lacks adequate healthy red blood cells.',
            'key_indicators': ['Low hemoglobin', 'Possible fatigue', 'Possible low blood pressure'],
            'recommendations': [
                'Consult a hematologist',
                'Get iron level tests',
                'Increase iron-rich foods',
                'Consider supplements if recommended'
            ]
        },
        'healthy': {
            'name': 'Healthy',
            'description': 'All indicators within normal ranges.',
            'key_indicators': ['Normal vital signs', 'Balanced health markers'],
            'recommendations': [
                'Maintain current healthy lifestyle',
                'Continue regular check-ups',
                'Stay physically active',
                'Eat a balanced diet'
            ]
        }
    }
    
    return disease_info.get(disease, {
        'name': disease,
        'description': 'Unknown condition',
        'key_indicators': [],
        'recommendations': ['Consult a healthcare professional']
    })