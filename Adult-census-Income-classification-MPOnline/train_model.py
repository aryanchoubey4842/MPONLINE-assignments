"""
Training Script — Adult Census Income Classification
Downloads the UCI Adult dataset, preprocesses it, trains a Random Forest classifier,
evaluates on a holdout test set, and saves the model pipeline + feature column info.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ── Column definitions ──────────────────────────────────────────────────────────
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Features to drop: fnlwgt (census weight, not predictive), education (redundant with education_num)
DROP_COLS = ['fnlwgt', 'education']

CATEGORICAL_FEATURES = [
    'workclass', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country'
]

NUMERICAL_FEATURES = [
    'age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
]

# ── Valid values for categorical features (used by the web app for dropdowns) ──
CATEGORICAL_OPTIONS = {
    'workclass': [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ],
    'marital_status': [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
        'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ],
    'occupation': [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ],
    'relationship': [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ],
    'race': [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ],
    'sex': ['Male', 'Female'],
    'native_country': [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
        'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
        'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
        'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
        'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
        'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
        'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
        'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'
    ]
}


def download_data():
    """
    Download the Adult dataset. Tries multiple sources in order:
    1. Local CSV file (if already downloaded)
    2. sklearn.datasets.fetch_openml (reliable, cached)
    3. Direct UCI ML Repository URLs (fallback)
    """
    local_csv = 'adult_dataset.csv'

    # ── Source 1: Local CSV ──────────────────────────────────────────────────
    if os.path.exists(local_csv):
        print(f"Loading cached dataset from {local_csv}...")
        df = pd.read_csv(local_csv)
        print(f"Loaded {len(df)} rows from local cache.")
        return df

    # ── Source 2: sklearn fetch_openml ────────────────────────────────────────
    try:
        from sklearn.datasets import fetch_openml
        print("Downloading via sklearn fetch_openml (OpenML #1590)...")
        data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
        df = data.frame

        # Rename columns to match our expected names
        rename_map = {
            'education-num': 'education_num',
            'marital-status': 'marital_status',
            'capital-gain': 'capital_gain',
            'capital-loss': 'capital_loss',
            'hours-per-week': 'hours_per_week',
            'native-country': 'native_country',
            'class': 'income'
        }
        df = df.rename(columns=rename_map)

        # Drop fnlwgt and education right away if present
        for col in ['fnlwgt', 'education']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Normalize income labels
        income_map = {'<=50K': '<=50K', '<=50K.': '<=50K',
                      '>50K': '>50K', '>50K.': '>50K'}
        df['income'] = df['income'].map(income_map).fillna(df['income'])

        # Cache locally for future runs
        df.to_csv(local_csv, index=False)
        print(f"Downloaded {len(df)} rows. Cached to {local_csv}.")
        return df

    except Exception as e:
        print(f"fetch_openml failed ({e}), trying direct UCI download...")

    # ── Source 3: Direct UCI URLs (fallback) ──────────────────────────────────
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    print("Downloading training data from UCI...")
    df_train = pd.read_csv(train_url, header=None, names=COLUMN_NAMES,
                           na_values=' ?', skipinitialspace=True)

    print("Downloading test data from UCI...")
    df_test = pd.read_csv(test_url, header=None, names=COLUMN_NAMES,
                          na_values=' ?', skipinitialspace=True, skiprows=1)

    # The test file has a trailing period on the income label — strip it
    df_test['income'] = df_test['income'].str.rstrip('.')

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Cache locally
    df.to_csv(local_csv, index=False)
    print(f"Combined dataset: {len(df)} rows. Cached to {local_csv}.")
    return df


def preprocess(df):
    """Clean the dataset and prepare features + target."""
    print("Preprocessing data...")

    # Drop rows with missing values (~7% of data)
    initial_len = len(df)
    df = df.dropna()
    print(f"  Dropped {initial_len - len(df)} rows with missing values "
          f"({len(df)} remaining)")

    # Drop uninformative columns (may already be dropped by download_data)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Encode target: <=50K → 0, >50K → 1
    df['income'] = (df['income'] == '>50K').astype(int)

    X = df.drop('income', axis=1)
    y = df['income']

    print(f"  Features: {X.shape[1]}  |  Class distribution: "
          f"<=50K = {(y == 0).sum()}, >50K = {(y == 1).sum()}")
    return X, y


def build_pipeline():
    """Build a sklearn Pipeline with preprocessing + Random Forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline


def train_and_save():
    """Main training orchestrator."""
    # 1. Get data
    df = download_data()

    # 2. Preprocess
    X, y = preprocess(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 4. Build & train pipeline
    pipeline = build_pipeline()
    print("\nTraining Random Forest (200 trees)...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

    # 6. Save artifacts
    model_path = 'model.pkl'
    columns_path = 'columns.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model pipeline saved to {model_path}")

    # Save column info + categorical options for the web app
    columns_info = {
        'feature_names': list(X.columns),
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_options': CATEGORICAL_OPTIONS
    }
    with open(columns_path, 'wb') as f:
        pickle.dump(columns_info, f)
    print(f"Column info saved to {columns_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    train_and_save()
