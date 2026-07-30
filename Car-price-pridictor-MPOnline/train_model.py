import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def generate_mock_data(n_samples=500):
    np.random.seed(42)
    # Generate mock car data
    years = np.random.randint(2010, 2024, n_samples)
    mileage = np.random.randint(10000, 150000, n_samples)
    engine_size = np.random.choice([1.0, 1.2, 1.4, 1.5, 1.6, 2.0, 2.5, 3.0, 3.5], n_samples)
    horsepower = np.random.randint(70, 400, n_samples)
    
    # Base price calculation (very simplified heuristic)
    # Base price goes up with year, engine size, horsepower, and goes down with mileage
    base_price = 10000
    price = (
        base_price + 
        (years - 2000) * 1000 + 
        engine_size * 2000 + 
        horsepower * 50 - 
        mileage * 0.1
    )
    # Add some noise
    price += np.random.normal(0, 2000, n_samples)
    price = np.maximum(price, 2000) # Ensure price is at least 2000
    
    df = pd.DataFrame({
        'Year': years,
        'Mileage': mileage,
        'EngineSize': engine_size,
        'Horsepower': horsepower,
        'Price': price
    })
    return df

def train_and_save_model():
    print("Generating mock car data...")
    df = generate_mock_data(1000)
    
    X = df[['Year', 'Mileage', 'EngineSize', 'Horsepower']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model trained successfully. R^2 Score: {score:.4f}")
    
    # Save the model
    model_path = 'model.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()
