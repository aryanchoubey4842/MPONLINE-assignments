# Adult Census Income Classification

Binary classification project that predicts whether a person's annual income exceeds **$50K** using the [UCI Adult Census dataset](https://archive.ics.uci.edu/ml/datasets/adult) (1994 Census data).

## Tech Stack

- **Model**: Random Forest Classifier (scikit-learn Pipeline)
- **Web Framework**: Flask
- **Dataset**: UCI Adult / Census Income (48,842 records, 14 attributes)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This downloads the dataset from UCI, preprocesses it, trains the model, and saves `model.pkl` + `columns.pkl`.

### 3. Run the web app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Age of the individual |
| workclass | Categorical | Employment type (Private, Govt, Self-emp, etc.) |
| education_num | Numerical | Years of education (1–16) |
| marital_status | Categorical | Marital status |
| occupation | Categorical | Occupation type |
| relationship | Categorical | Family role (Husband, Wife, Own-child, etc.) |
| race | Categorical | Race |
| sex | Categorical | Male / Female |
| capital_gain | Numerical | Investment income ($) |
| capital_loss | Numerical | Investment losses ($) |
| hours_per_week | Numerical | Weekly working hours |
| native_country | Categorical | Country of origin |

**Target**: Income — `<=50K` (0) or `>50K` (1)

## Project Structure

```
├── train_model.py      # Data download, preprocessing, training, evaluation
├── app.py              # Flask web application
├── templates/
│   └── index.html      # Prediction form UI
├── static/
│   └── style.css       # Stylesheet
├── model.pkl           # Trained model pipeline (generated)
├── columns.pkl         # Feature metadata (generated)
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
└── README.md
```

## Deployment

Configured for [Render](https://render.com) via `render.yaml`. Push to a connected Git repo and Render will auto-deploy.

## License

Dataset provided by UCI Machine Learning Repository. Project created for educational purposes.
