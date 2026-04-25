from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import uuid
from bson import ObjectId
import os

app = Flask(__name__)
CORS(app)

# ---------- Local MongoDB Connection ----------
MONGODB_URI = "mongodb://localhost:27017/"

try:
    client = MongoClient(MONGODB_URI)
    db = client['internshield_db']
    checks_collection = db['internship_checks']
    behavioral_collection = db['behavioral_checks']
    
    client.admin.command('ping')
    print("="*60)
    print("✅ Connected to LOCAL MongoDB successfully!")
    print("📍 MongoDB is running on: localhost:27017")
    print("="*60)
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    print("\n⚠️ Please make sure MongoDB is running")
    exit(1)

# ---------- Train ML Model ----------
print("\n🔄 Training Machine Learning Model...")

training_descriptions = [
    "Pay $100 registration fee to apply for internship",
    "Immediate hiring work from home earn 5000 weekly deposit required",
    "Guaranteed placement after paying certification fee",
    "Upfront payment for training materials required",
    "Cryptocurrency internship earn thousands without experience",
    "Need to pay for background verification",
    "Unlimited earning potential pay $50 to start",
    "No interview required join immediately with registration fee",
    "Get rich quick internship pay for premium access",
    "Urgent opening need to pay for visa processing",
    "Work from home data entry earn 20000 per month pay security",
    "Easy money fast track career just pay $20",
    "Send us $100 for application processing",
    "Pay for your own laptop and we'll reimburse later",
    "Investment required for training program",
    "Paid software engineering internship at Google with stipend",
    "Marketing internship with mentorship and health insurance",
    "Data science intern role competitive salary and learning opportunities",
    "Official summer internship program with professional development",
    "Frontend developer internship paid stipend and flexible hours",
    "Research assistant internship at university scholarship provided",
    "Finance internship with training and career growth",
    "Design internship portfolio building and paid work",
    "Legal internship with experienced attorneys and stipend",
    "Nonprofit internship with meaningful work and compensation",
    "Engineering intern position with structured mentorship",
    "Content writing internship competitive stipend",
    "Sales internship with base salary plus commission",
    "Product management intern role with real projects",
    "Cloud computing internship with certification sponsorship"
]

training_labels = [1]*15 + [0]*15

vectorizer = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
X_train = vectorizer.fit_transform(training_descriptions)
ml_model = LogisticRegression(random_state=42)
ml_model.fit(X_train, training_labels)

print("✅ ML Model trained successfully!")

SCAM_KEYWORDS = [
    'fee', 'deposit', 'pay', 'urgent', 'immediate start', 'no interview',
    'any qualification', 'registration', 'cryptocurrency', 'upfront',
    'certification', 'easy money', 'guaranteed placement', 'refund',
    'background verification', 'processing fee', 'invest', 'security deposit',
    'quick money', 'get rich', 'weekly payment', 'no experience required'
]

def predict_scam_probability(text):
    """Predict scam probability using ML model"""
    if not text or len(text.strip()) < 10:
        return 50.0, False
    
    X_new = vectorizer.transform([text.lower()])
    probability = ml_model.predict_proba(X_new)[0]
    # Convert numpy float to Python float
    scam_prob = float(round(probability[1] * 100, 2))
    is_scam = scam_prob > 50
    return scam_prob, is_scam

def extract_scam_keywords(text):
    """Extract scam keywords from text"""
    found = []
    text_lower = text.lower()
    for keyword in SCAM_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return list(set(found[:10]))

# ---------- API Routes ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/check-internship', methods=['POST'])
def check_internship():
    try:
        data = request.json
        session_id = data.get('session_id')
        company = data.get('company', '')
        title = data.get('title', '')
        description = data.get('description', '')
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        full_text = f"{company} {title} {description}"
        
        # Get ML prediction
        scam_prob, is_scam = predict_scam_probability(full_text)
        
        # Extract suspicious keywords
        keywords = extract_scam_keywords(full_text)
        
        # Convert numpy bool to Python bool
        is_scam_python = bool(is_scam)
        
        # Store in MongoDB (all Python native types)
        record = {
            'session_id': str(session_id),
            'type': 'internship',
            'timestamp': datetime.now(),
            'company': str(company),
            'title': str(title),
            'description': str(description[:500]),
            'scam_probability': float(scam_prob),
            'is_scam': is_scam_python,
            'keywords_found': keywords,
            'full_text_length': int(len(full_text))
        }
        
        checks_collection.insert_one(record)
        
        if is_scam_python:
            message = "⚠️ HIGH RISK: This appears to be a scam internship!"
            recommendation = "🚨 Avoid this internship. Never pay money for a job opportunity."
        else:
            message = "✅ LOW RISK: This internship looks legitimate"
            recommendation = "✓ Proceed with normal caution. Still do your own research."
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'scam_probability': scam_prob,
            'is_scam': is_scam_python,
            'keywords_found': keywords,
            'message': message,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/behavioral-check', methods=['POST'])
def behavioral_check():
    try:
        data = request.json
        session_id = data.get('session_id')
        answers = data.get('answers', [])
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        score_map = {'Yes': 20, 'Maybe': 10, 'No': 0}
        total_score = sum(score_map.get(ans, 0) for ans in answers)
        behavioral_risk = int(min(100, total_score))  # Convert to Python int
        
        if behavioral_risk > 60:
            risk_level = "🔴 HIGH RISK"
            advice = "Strongly reconsider - multiple red flags detected"
        elif behavioral_risk > 30:
            risk_level = "🟡 MEDIUM RISK"
            advice = "Proceed with caution - investigate further"
        else:
            risk_level = "🟢 LOW RISK"
            advice = "Looks good, but always stay vigilant"
        
        record = {
            'session_id': str(session_id),
            'type': 'behavioral',
            'timestamp': datetime.now(),
            'answers': [str(a) for a in answers],  # Convert all to strings
            'behavioral_risk': behavioral_risk,
            'risk_level': str(risk_level),
            'advice': str(advice)
        }
        
        behavioral_collection.insert_one(record)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'behavioral_risk': behavioral_risk,
            'risk_level': risk_level,
            'advice': advice,
            'message': f'Behavioral Risk Score: {behavioral_risk}/100 - {risk_level}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'success': True, 'checks': []})
        
        internship_checks = list(checks_collection.find(
            {'session_id': session_id}
        ).sort('timestamp', -1))
        
        behavioral_checks = list(behavioral_collection.find(
            {'session_id': session_id}
        ).sort('timestamp', -1))
        
        for check in internship_checks:
            check['_id'] = str(check['_id'])
            if check.get('timestamp'):
                check['timestamp'] = check['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            # Ensure all numpy types are converted
            if 'scam_probability' in check:
                check['scam_probability'] = float(check['scam_probability'])
            if 'is_scam' in check:
                check['is_scam'] = bool(check['is_scam'])
        
        for check in behavioral_checks:
            check['_id'] = str(check['_id'])
            if check.get('timestamp'):
                check['timestamp'] = check['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            if 'behavioral_risk' in check:
                check['behavioral_risk'] = int(check['behavioral_risk'])
        
        all_checks = internship_checks + behavioral_checks
        all_checks.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'checks': all_checks,
            'total_checks': len(all_checks)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️ INTERNSHIELD SERVER STARTING")
    print("="*60)
    print("📍 Web Application: http://127.0.0.1:5000")
    print("🤖 ML Model Status: ACTIVE")
    print("💾 Database: Local MongoDB")
    print("="*60)
    print("\n✅ Ready to detect scam internships!\n")
    app.run(debug=True, host='127.0.0.1', port=5000)