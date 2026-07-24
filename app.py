import os
import joblib
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load Model
MODEL_PATH = "decision_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: '{MODEL_PATH}' not found in current directory.")

# Embedded HTML Template with inline CSS Styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Predictor AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(148, 163, 184, 0.2);
            --input-focus: #818cf8;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 520px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            width: 100%;
            padding: 0.85rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 1rem;
            transition: all 0.3s ease;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--input-focus);
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.2);
        }

        select option {
            background-color: #1e293b;
            color: #f8fafc;
        }

        .submit-btn {
            width: 100%;
            padding: 1rem;
            margin-top: 1rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .submit-btn:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 16px;
            text-align: center;
            font-weight: 600;
            animation: fadeIn 0.4s ease;
        }

        .result-card.yes {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--success-color);
            color: #34d399;
        }

        .result-card.no {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger-color);
            color: #f87171;
        }

        .result-card.error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger-color);
            color: #f87171;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Decision Model</h1>
            <p>Input profile details to predict outcome</p>
        </div>

        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" placeholder="e.g. 35" value="{{ form_data.age if form_data else '' }}" required min="1" max="120">
            </div>

            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="" disabled {{ 'selected' if not form_data }}>Select Gender</option>
                    <option value="0" {{ 'selected' if form_data and form_data.gender == '0' }}>Female (0)</option>
                    <option value="1" {{ 'selected' if form_data and form_data.gender == '1' }}>Male (1)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="region">Region Code</label>
                <input type="number" id="region" name="region" placeholder="e.g. 0, 1, 2" value="{{ form_data.region if form_data else '' }}" required>
            </div>

            <div class="form-group">
                <label for="occupation">Occupation Code</label>
                <input type="number" id="occupation" name="occupation" placeholder="e.g. 0, 1, 2" value="{{ form_data.occupation if form_data else '' }}" required>
            </div>

            <div class="form-group">
                <label for="income">Income</label>
                <input type="number" step="any" id="income" name="income" placeholder="e.g. 50000" value="{{ form_data.income if form_data else '' }}" required>
            </div>

            <button type="submit" class="submit-btn">Predict Outcome</button>
        </form>

        {% if prediction %}
            <div class="result-card {{ prediction.lower() }}">
                Prediction Result: {{ prediction | upper }}
            </div>
        {% endif %}

        {% if error %}
            <div class="result-card error">
                Error: {{ error }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None, error=None, form_data=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=None, 
            error="Model file 'decision_model.pkl' not found on server.",
            form_data=request.form
        )

    try:
        age = float(request.form.get("age"))
        gender = float(request.form.get("gender"))
        region = float(request.form.get("region"))
        occupation = float(request.form.get("occupation"))
        income = float(request.form.get("income"))

        # Features order matching: ['Age', 'Gender', 'Region', 'Occupation', 'Income']
        features = np.array([[age, gender, region, occupation, income]])
        
        pred = model.predict(features)[0]

        return render_template_string(
            HTML_TEMPLATE, 
            prediction=str(pred), 
            error=None,
            form_data=request.form
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=None, 
            error=str(e),
            form_data=request.form
        )

if __name__ == "__main__":
    app.run(debug=True)
