import os
import subprocess
import glob
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Ensure videos directory exists
VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

import sys

@app.route("/run-simulation", methods=["POST"])
def run_simulation():
    try:
        # Run the record_video.py script to generate a new video
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_video.py")
        subprocess.run([sys.executable, script_path], check=True)
        return jsonify({"status": "success", "message": "Simulation completed."})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/latest-video")
def latest_video():
    # Find the most recently created mp4 file in the videos directory
    list_of_files = glob.glob(os.path.join(VIDEOS_DIR, '*.mp4'))
    if not list_of_files:
        return jsonify({"status": "error", "message": "No video found"}), 404
    
    latest_file = max(list_of_files, key=os.path.getctime)
    filename = os.path.basename(latest_file)
    
    # Return the video URL path
    return jsonify({"status": "success", "video_url": f"/videos/{filename}"})

@app.route("/videos/<filename>")
def serve_video(filename):
    return send_from_directory(VIDEOS_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
