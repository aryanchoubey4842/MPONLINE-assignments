from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from recommender import Recommender
from pydantic import BaseModel
import os

app = FastAPI(title="Movie Recommendation API")

# Initialize recommender
data_path = os.path.join(os.path.dirname(__file__), "data", "tmdb_5000_movies.csv")
recommender = Recommender(data_path)

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/recommend")
async def get_recommendation(title: str = Query(..., description="Title of the movie")):
    recommendations = recommender.get_recommendations(title)
    return {"title": title, "recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
