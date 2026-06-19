from flask import Flask, render_template, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import uuid
from pathlib import Path
from datetime import datetime

from genetic_planner import TreatmentGeneticAlgorithm
from openai import OpenAI

app = Flask(__name__)

# ==========================================
# SAAS DATABASE CONFIG
# ==========================================
HISTORY_FILE = 'patient_history.json'
USERS_FILE = 'users.json'

def load_json(filepath):
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r') as f:
        try: return json.load(f)
        except json.JSONDecodeError: return {}

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

if not os.path.exists(HISTORY_FILE): save_json(HISTORY_FILE, {})
if not os.path.exists(USERS_FILE): save_json(USERS_FILE, {})

# ==========================================
# LLM CHATBOT CONFIGURATION
# ==========================================
GROQ_API_KEY = "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" 
try:
    ai_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
except Exception as e:
    ai_client = None

# ==========================================
# MACHINE LEARNING ARTIFACTS
# ==========================================
MODEL_DIR = Path('models')
try:
    nb_model = joblib.load(MODEL_DIR / 'nb_model.pkl')
    rf_model = joblib.load(MODEL_DIR / 'rf_model.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    label_encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    print("✓ Naive Bayes, Random Forest, and Scalers loaded successfully!")
except Exception as e:
    print(f"✗ Error loading ML artifacts: {e}")
    nb_model = rf_model = scaler = label_encoder = None

# Added 'gender' to feature columns
FEATURE_COLUMNS = ['gender', 'glucose', 'blood_pressure', 'heart_rate', 'hemoglobin', 'cholesterol', 'bmi', 'age']

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def parse_report(report_text):
    data = {}
    for line in report_text.replace(',', '\n').split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            k = key.strip().lower().replace(' ', '_')
            v = value.strip()
            if k == 'gender':
                data[k] = 1.0 if v.lower() == 'male' else 0.0
            else:
                try: data[k] = float(v)
                except ValueError: continue
    return data

def parse_csv_data(file):
    try:
        content = file.read().decode('utf-8', errors='ignore')
        if ':' in content.split('\n')[0]:
            data = {}
            for line in content.split('\n'):
                line = line.strip().strip('"').strip("'")  
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    if key == 'gender':
                        data[key] = 1.0 if value.strip().lower() == 'male' else 0.0
                    else:
                        if key == 'blood_pressure' and '/' in value:
                            value = value.split('/')[0]
                        try: data[key] = float(value.strip())
                        except ValueError: continue
            return data
            
        df = pd.read_csv(io.StringIO(content))
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df.rename(columns={'bp': 'blood_pressure', 'hr': 'heart_rate', 'chol': 'cholesterol'}, inplace=True)
        
        raw_data = df.iloc[0].to_dict()
        clean_data = {}
        for key, val in raw_data.items():
            if key == 'gender':
                clean_data[key] = 1.0 if str(val).lower() == 'male' else 0.0
            else:
                try:
                    if key == 'blood_pressure' and isinstance(val, str) and '/' in val:
                        clean_data[key] = float(val.split('/')[0])
                    else: clean_data[key] = float(val)
                except (ValueError, TypeError): pass 
                
        if not any(k in clean_data for k in FEATURE_COLUMNS):
            raise ValueError("No recognizable medical columns found in the CSV.")
        return clean_data
    except Exception as e:
        raise ValueError(f"Parsing Failed: {str(e)}")

def prepare_features(data):
    features = []
    for col in FEATURE_COLUMNS:
        val = data.get(col, 0.0)
        if col == 'gender' and isinstance(val, str):
            val = 1.0 if val.lower() == 'male' else 0.0
        features.append(val)
        
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    if scaler is not None: return scaler.transform(feature_df)
    return feature_df.values

# ==========================================
# AUTHENTICATION ROUTES
# ==========================================
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    users = load_json(USERS_FILE)
    email = data.get('email', '').lower()
    
    if email in users: 
        return jsonify({'error': 'Email already registered.'}), 400
        
    # Save age and gender securely to the database
    users[email] = {
        'name': data.get('name'), 
        'password': generate_password_hash(data.get('password')),
        'age': data.get('age'),
        'gender': data.get('gender')
    }
    save_json(USERS_FILE, users)
    
    return jsonify({
        'success': True, 
        'name': data.get('name'), 
        'email': email,
        'age': data.get('age'),
        'gender': data.get('gender')
    })

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    users = load_json(USERS_FILE)
    email = data.get('email', '').lower()
    
    if email in users and check_password_hash(users[email]['password'], data.get('password')):
        user_data = users[email]
        # Return age and gender to auto-fill the frontend dashboard
        return jsonify({
            'success': True, 
            'name': user_data.get('name'), 
            'email': email,
            'age': user_data.get('age'),
            'gender': user_data.get('gender')
        })
        
    return jsonify({'error': 'Invalid email or password.'}), 401

# ==========================================
# CORE ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse_csv', methods=['POST'])
def parse_csv():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        data = parse_csv_data(file)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    if nb_model is None or rf_model is None: 
        return jsonify({'error': 'Machine Learning models not loaded. Check backend.'}), 500
        
    try:
        patient_email = request.form.get('patient_email')
        data = parse_report(request.form.get('report_text', ''))
        
        if not data: return jsonify({'error': 'Could not parse report data'}), 400
        
        scaled_features = prepare_features(data)
        
        pred_encoded = rf_model.predict(scaled_features)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        
        nb_proba = nb_model.predict_proba(scaled_features)[0]
        prob_breakdown = {}
        
        try:
            for idx, prob in zip(nb_model.classes_, nb_proba):
                disease_name = label_encoder.inverse_transform([idx])[0]
                prob_breakdown[disease_name.lower()] = float(prob) * 100.0
                
            age_val = data.get('age', 0)
            if age_val >= 50:
                age_risk = (age_val - 50) * 0.8 
                if 'healthy' in prob_breakdown:
                    reduction = min(prob_breakdown['healthy'] - 2.0, age_risk * 2)
                    if reduction > 0:
                        prob_breakdown['healthy'] -= reduction
                        prob_breakdown['heart'] = prob_breakdown.get('heart', 0) + (reduction * 0.6)
                        prob_breakdown['diabetes'] = prob_breakdown.get('diabetes', 0) + (reduction * 0.4)
            
            total_prob = sum(prob_breakdown.values())
            for k in prob_breakdown:
                prob_breakdown[k] = round((prob_breakdown[k] / total_prob) * 100.0, 1)

            pred_class_index = label_encoder.transform([prediction])[0]
            confidence = prob_breakdown.get(prediction.lower(), float(nb_proba[pred_class_index]) * 100) / 100.0

        except Exception:
            confidence = float(max(nb_proba))
            prob_breakdown = {'healthy': 0, 'diabetes': 0, 'heart': 0, 'anemia': 0}
            prob_breakdown[prediction.lower()] = 100.0
    
        db = load_json(HISTORY_FILE)
        if patient_email not in db: db[patient_email] = []
            
        patient_records = db[patient_email]
        deltas = {}
        if len(patient_records) > 0:
            last_record = patient_records[-1]
            deltas = {
                'glucose': round(data.get('glucose', 0) - last_record.get('glucose', 0), 1),
                'bmi': round(data.get('bmi', 0) - last_record.get('bmi', 0), 1),
                'cholesterol': round(data.get('cholesterol', 0) - last_record.get('cholesterol', 0), 1),
                'bp_systolic': round(data.get('blood_pressure', 0) - last_record.get('bp_systolic', 0), 1)
            }

        ga_planner = TreatmentGeneticAlgorithm(diagnosis=prediction.lower(), bmi=data.get('bmi', 22.0), target_calories=2000)
        generated_plan = ga_planner.generate_plan(generations=50)

        new_record = {
            'id': str(uuid.uuid4()),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'age': data.get('age', 0),
            'glucose': data.get('glucose', 0),
            'bmi': data.get('bmi', 0),
            'cholesterol': data.get('cholesterol', 0),
            'bp_systolic': data.get('blood_pressure', 0),
            'hemoglobin': data.get('hemoglobin', 0),
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_breakdown, 
            'treatment_plan': generated_plan,
            'completed_days': [] 
        }
        db[patient_email].append(new_record)
        save_json(HISTORY_FILE, db)
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence * 100:.1f}%",
            'probabilities': prob_breakdown,
            'data': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in data.items()},
            'deltas': deltas,
            'treatment_plan': generated_plan,
            'completed_days': []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard_data', methods=['POST'])
def dashboard_data():
    patient_email = request.json.get('patient_email')
    db = load_json(HISTORY_FILE)
    
    if patient_email not in db or len(db[patient_email]) == 0:
        return jsonify({'empty': True})

    records = db[patient_email]
    dates = [r['date'].split(' ')[0] for r in records]
    
    glucose_vals = [r.get('glucose', 0) for r in records]
    bmi_vals = [r.get('bmi', 0) for r in records]
    chol_vals = [r.get('cholesterol', 0) for r in records]
    bp_vals = [r.get('bp_systolic', 0) for r in records]
    
    improvements, risks = [], []
    if len(records) > 1:
        last, prev = records[-1], records[-2]
        
        g_diff = last.get('glucose',0) - prev.get('glucose',0)
        if g_diff < -2: improvements.append(f"Glucose dropped by {abs(g_diff):.1f} mg/dL")
        elif g_diff > 5: risks.append(f"Glucose elevated by {g_diff:.1f} mg/dL")
            
        b_diff = last.get('bmi',0) - prev.get('bmi',0)
        if b_diff < -0.5: improvements.append(f"BMI reduced by {abs(b_diff):.1f}")
        elif b_diff > 0.5: risks.append(f"BMI increased by {b_diff:.1f}")
            
        c_diff = last.get('cholesterol',0) - prev.get('cholesterol',0)
        if c_diff < -5: improvements.append(f"Cholesterol improved by {abs(c_diff):.1f} mg/dL")
        elif c_diff > 10: risks.append(f"Cholesterol spiked by {c_diff:.1f} mg/dL")

    latest_report = records[-1]
    age = latest_report.get('age', 0)
    pred = latest_report.get('prediction', '').lower()
    
    if age > 50 and pred == 'heart':
        risks.append(f"Age Correlation: Advanced age ({age}) is acting as a mathematical multiplier for cardiovascular risk.")
    elif age > 45 and pred == 'diabetes':
        risks.append(f"Age Correlation: Reduced metabolic clearance at age {age} correlates heavily with your glycemic risk.")
    elif age > 60 and pred == 'healthy':
        improvements.append(f"Age Correlation: Excellent vitality! Maintaining healthy biomarkers at age {age} is statistically exceptional.")

    trendlines = {'glucose': [], 'bmi': [], 'cholesterol': [], 'bp_systolic': []}
    
    def get_trend(vals):
        if len(vals) > 1:
            X = np.array(range(len(vals))).reshape(-1, 1)
            y = np.array(vals)
            reg = LinearRegression().fit(X, y)
            return np.round(reg.predict(np.array(range(len(vals) + 1)).reshape(-1, 1)), 1).tolist()
        return []

    trendlines['glucose'] = get_trend(glucose_vals)
    trendlines['bmi'] = get_trend(bmi_vals)
    trendlines['cholesterol'] = get_trend(chol_vals)
    trendlines['bp_systolic'] = get_trend(bp_vals)
    
    if len(dates) > 1: dates.append("AI Projection")
    
    return jsonify({
        'empty': False, 
        'dates': dates, 
        'metrics': {
            'glucose': glucose_vals, 'bmi': bmi_vals,
            'cholesterol': chol_vals, 'bp_systolic': bp_vals
        },
        'trendlines': trendlines,
        'history': list(reversed(records)),
        'improvements': improvements, 
        'risks': risks,
        'latest': {
            'glucose': latest_report.get('glucose'),
            'bmi': latest_report.get('bmi'),
            'cholesterol': latest_report.get('cholesterol'),
            'prediction': latest_report.get('prediction', 'Unknown').upper(),
            'probabilities': latest_report.get('probabilities', {}),
            'treatment_plan': latest_report.get('treatment_plan', []),
            'completed_days': latest_report.get('completed_days', [])
        }
    })

@app.route('/api/update_progress', methods=['POST'])
def update_progress():
    data = request.json
    email = data.get('patient_email')
    completed_days = data.get('completed_days', [])
    
    db = load_json(HISTORY_FILE)
    if email in db and len(db[email]) > 0:
        db[email][-1]['completed_days'] = completed_days
        save_json(HISTORY_FILE, db)
    return jsonify({'success': True})

@app.route('/api/delete_history', methods=['POST'])
def delete_history():
    data = request.json
    email = data.get('patient_email')
    record_id = data.get('record_id')
    
    db = load_json(HISTORY_FILE)
    if email in db:
        db[email] = [r for r in db[email] if r.get('id') != record_id and r.get('date') != record_id]
        save_json(HISTORY_FILE, db)
        return jsonify({'success': True})
    return jsonify({'error': 'Record not found'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        user_message = request.json.get('message', '')
        file_context = ""
    else:
        user_message = request.form.get('message', '')
        file_context = ""
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                try:
                    file_content = file.read().decode('utf-8', errors='ignore')
                    max_chars = 3000 
                    if len(file_content) > max_chars:
                        file_content = file_content[:max_chars] + "\n...[Content truncated]"
                    file_context = f"\n\n[CONTEXT: The user uploaded a file named '{file.filename}'. Here are its contents:\n{file_content}\n]"
                except Exception as e:
                    file_context = f"\n\n[System Note: The user tried to upload a file, but it could not be read. Error: {str(e)}]"

    system_prompt = """
    You are HealthSense AI, a highly professional, empathetic, and knowledgeable clinical assistant. 
    Your rules:
    1. Keep responses concise (under 6-7 lines) and easy to read.
    2. Explain medical terminology clearly.
    3. If the user uploads a file, use the file contents provided in the prompt to answer their questions.
    4. CRITICAL: Always include a brief disclaimer that you are an AI assistant.
    5. STRICT DOMAIN RESTRICTION: You are strictly a medical assistant. If the user asks non-medical questions, politely decline.
    6. DOMAIN ENFORCEMENT: If a user asks a question UNRELATED to healthcare, medicine, or wellness, YOU MUST REFUSE TO ANSWER. 
    7. PROHIBITED TOPICS: Do NOT write code (Python, HTML, etc.). Do NOT write essays. Do NOT answer math, history, politics, or general trivia questions. Do NOT translate languages unless it is a medical text.
    8. MANDATORY FALLBACK: If a user attempts to ask an out-of-domain question, you must stop generation and reply EXACTLY with this phrase: "I am HealthSense AI, a clinical assistant. I am strictly programmed to only assist with medical, health, and wellness inquiries. How can I help you with your health today?"
    """
    
    full_prompt = user_message + file_context

    try:
        response = ai_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7, 
            max_tokens=500
        )
        return jsonify({'reply': response.choices[0].message.content.replace('**', '')})
    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        return jsonify({'reply': "I am currently experiencing network latency or an error occurred."})

if __name__ == '__main__':
    if not os.path.exists(HISTORY_FILE): save_json(HISTORY_FILE, {})
    if not os.path.exists(USERS_FILE): save_json(USERS_FILE, {})
    MODEL_DIR.mkdir(exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)