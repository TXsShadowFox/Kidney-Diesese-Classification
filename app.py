from flask import Flask, request, jsonify, render_template
import os
import base64
import binascii
import hmac
import subprocess
import sys
import threading
from pathlib import Path
from flask_cors import CORS, cross_origin
from PIL import UnidentifiedImageError
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.common import load_json



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

training_lock = threading.Lock()
training_process = None


class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()


# created at import time (the model loads lazily) so it also exists under gunicorn
clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    # test-set metrics written by the evaluation stage; the page hides its model report without them
    scores_path = Path("scores.json")
    scores = load_json(scores_path) if scores_path.exists() else None
    return render_template('index.html', scores=scores)




@app.route("/train", methods=['POST'])
@cross_origin()
def trainRoute():
    global training_process
    # disabled unless TRAIN_TOKEN is set; callers send it in the X-Train-Token header
    expected_token = os.environ.get("TRAIN_TOKEN")
    if not expected_token:
        return "Training endpoint is disabled", 403
    given_token = request.headers.get("X-Train-Token", "")
    if not hmac.compare_digest(given_token.encode(), expected_token.encode()):
        return "Unauthorized", 401

    with training_lock:
        if training_process is not None and training_process.poll() is None:
            return "Training already running", 409
        # run in the background so the request returns right away
        training_process = subprocess.Popen([sys.executable, "main.py"])
    return "Training started", 202



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.get_json(silent=True) or {}
    image = data.get('image')
    if not isinstance(image, str):
        return jsonify({"error": "Send JSON with a base64-encoded 'image' field"}), 400
    # each request decodes its own bytes in memory, so concurrent requests can't mix up images
    try:
        result = clApp.classifier.predict(base64.b64decode(image, validate=True))
    except (binascii.Error, UnidentifiedImageError):
        return jsonify({"error": "Could not read the uploaded image"}), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) #for AWS