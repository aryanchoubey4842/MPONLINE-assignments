# Movie Recommendation System

A full-stack movie recommendation application built with **FastAPI**, **scikit-learn**, and a modern **HTML/CSS/JS** frontend.

## Overview
This system uses a **Content-Based Filtering** algorithm to recommend movies based on a user's favorite movie. It uses the TMDB 5,000 Movies dataset and analyzes the `genres` and `overview` (plot summary) using TF-IDF and Cosine Similarity to find the most relevant matches.

## Features
- **Modern UI**: A responsive, dark-themed frontend utilizing glassmorphism and subtle animations for a premium user experience.
- **FastAPI Backend**: A lightweight, asynchronous python backend that processes data and serves the frontend.
- **Machine Learning**: `scikit-learn` powers the TF-IDF Vectorizer and linear kernel similarity calculations.
- **Dockerized**: A complete `Dockerfile` is included for easy containerization and deployment to services like Render.

## Getting Started Locally

### Prerequisites
- Python 3.10+
- (Optional) Docker

### Native Python Setup
1. Clone the repository and navigate into the project directory.
2. (Recommended) Create and activate a virtual environment.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the development server:
   ```bash
   python app.py
   ```
   *(Or alternatively run: `uvicorn app:app --reload`)*
5. Open your browser and navigate to `http://127.0.0.1:8000`

### Docker Setup
1. Build the Docker image:
   ```bash
   docker build -t movie-recommender .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 movie-recommender
   ```
3. Open your browser and navigate to `http://localhost:8000`

## Dataset
The dataset utilized is the popular [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Ensure that the `tmdb_5000_movies.csv` file is located in the `data/` directory.

## Deployment (Render)
A `render.yaml` configuration is included. You can connect your GitHub repository to Render and create a new **Web Service** using the Docker environment setup automatically.

## License
MIT License
