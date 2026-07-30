FROM python:3.10-slim

# Install system dependencies required for Gymnasium, Box2D, and MoviePy
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    ffmpeg \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable to allow pygame to render headlessly without a display
ENV SDL_VIDEODRIVER=dummy

# Expose the Flask port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
