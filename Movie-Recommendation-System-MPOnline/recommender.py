import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast
import os
import difflib

class Recommender:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.movies_df = pd.DataFrame()
        self.cosine_sim = None
        self.indices = None
        self.load_data()

    def parse_genres(self, x):
        try:
            genres_list = ast.literal_eval(x)
            return "|".join([i['name'] for i in genres_list])
        except (ValueError, TypeError):
            return ""

    def load_data(self):
        if os.path.exists(self.data_path):
            self.movies_df = pd.read_csv(self.data_path)
            
            # Extract genres from JSON-like string
            self.movies_df['genre_str'] = self.movies_df['genres'].apply(self.parse_genres)
            self.movies_df['genre'] = self.movies_df['genre_str'] # For the frontend to use

            # Combine genre and overview for content-based filtering
            self.movies_df['content'] = self.movies_df['genre_str'] + " " + self.movies_df['overview'].fillna('')
            
            # TF-IDF Vectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
            
            # Compute Cosine Similarity
            self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            # Reverse mapping of indices and movie titles
            self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'].str.lower()).drop_duplicates()
        else:
            print(f"Data file {self.data_path} not found.")

    def get_recommendations(self, title: str, num_recommendations: int = 6):
        title_lower = title.lower()
        if self.indices is None:
            return {"error": "Movie database not loaded."}
            
        if title_lower not in self.indices:
            # Try finding a close match
            close_matches = difflib.get_close_matches(title_lower, self.indices.index.tolist(), n=1, cutoff=0.4)
            if not close_matches:
                # Also try substring matching
                substring_matches = [t for t in self.indices.index.tolist() if title_lower in t]
                if not substring_matches:
                    return {"error": "Movie not found"}
                title_lower = substring_matches[0]
            else:
                title_lower = close_matches[0]
        
        idx = self.indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip the movie itself (index 0)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = self.movies_df.iloc[movie_indices][['title', 'genre', 'overview']].to_dict(orient='records')
        return recommended_movies
