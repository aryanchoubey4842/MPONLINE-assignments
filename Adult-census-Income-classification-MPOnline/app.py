"""
Flask Web Application — Adult Census Income Classification
Loads the trained model pipeline and serves a web UI for income prediction.
"""

from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ── Load trained model and column info ───────────────────────────────────────────
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns_info = pickle.load(f)
    print("Model and column info loaded successfully.")
except FileNotFoundError:
    model = None
    columns_info = None
    print("Warning: model.pkl or columns.pkl not found. "
          "Please run train_model.py first.")


@app.route('/')
def home():
    """Render the prediction form."""
    return render_template('index.html', columns_info=columns_info)


@app.route('/predict', methods=['POST'])
def predict():
    """Process form input, run inference, and return the result."""
    if model is None or columns_info is None:
        return "Model not found. Please train the model first.", 500

    try:
        # Build a single-row DataFrame with the same columns the model expects
        input_data = {}

        # Numerical features
        input_data['age'] = int(request.form['age'])
        input_data['education_num'] = int(request.form['education_num'])
        input_data['capital_gain'] = int(request.form['capital_gain'])
        input_data['capital_loss'] = int(request.form['capital_loss'])
        input_data['hours_per_week'] = int(request.form['hours_per_week'])

        # Categorical features
        input_data['workclass'] = request.form['workclass']
        input_data['marital_status'] = request.form['marital_status']
        input_data['occupation'] = request.form['occupation']
        input_data['relationship'] = request.form['relationship']
        input_data['race'] = request.form['race']
        input_data['sex'] = request.form['sex']
        input_data['native_country'] = request.form['native_country']

        # Create DataFrame in the correct column order
        df_input = pd.DataFrame([input_data], columns=columns_info['feature_names'])

        # Predict
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        confidence = probabilities[prediction] * 100

        result = {
            'prediction': '>50K' if prediction == 1 else '≤50K',
            'confidence': f"{confidence:.1f}",
            'is_high_income': prediction == 1
        }

        return render_template('index.html',
                               columns_info=columns_info,
                               result=result,
                               form_data=input_data)

    except Exception as e:
        return f"Prediction error: {str(e)}", 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
