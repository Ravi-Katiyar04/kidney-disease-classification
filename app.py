import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS, cross_origin

# Hugging Face Hub model integration for deployment:
# from huggingface_hub import hf_hub_download
# from tensorflow.keras.models import load_model
#
# model_path = hf_hub_download(
#     repo_id="katiyar2002/kidney-disease-classification",
#     filename="model.h5"
# )
# model = load_model(model_path, compile=False)

from cnnClassifier import logger
from cnnClassifier.utils.common import decodeImage, load_json
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set locale environment variables
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        # PredictionPipeline downloads and caches the model from Hugging Face Hub for cloud deployment
        self.classifier = PredictionPipeline(self.filename)


# Instantiate at module level so it functions across all WSGI/ASGI and Flask runners
clApp = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    """Renders the Kidney Disease Classification Web UI."""
    return render_template("index.html")


@app.route("/metrics", methods=["GET"])
@cross_origin()
def get_metrics():
    """Returns the latest evaluation metrics from scores.json if available."""
    scores_file = Path("scores.json")
    if scores_file.exists():
        try:
            scores = load_json(scores_file)
            return jsonify({
                "loss": scores.get("loss", 0.0),
                "accuracy": scores.get("accuracy", 0.0),
                "model": "ResNet50V2",
                "classes": ["Cyst", "Normal", "Stone", "Tumor"]
            })
        except Exception as e:
            logger.warning(f"Could not read scores.json: {e}")
    return jsonify({
        "loss": 0.6338,
        "accuracy": 0.8440,
        "model": "ResNet50V2",
        "classes": ["Cyst", "Normal", "Stone", "Tumor"]
    })


@app.route("/sample/<category>", methods=["GET"])
@cross_origin()
def get_sample_image(category: str):
    """Serves a real CT scan sample from the dataset for testing."""
    valid_categories = {"Cyst", "Normal", "Stone", "Tumor"}
    cat_normalized = category.capitalize()
    if cat_normalized not in valid_categories:
        return jsonify({"error": f"Invalid category. Must be one of {valid_categories}"}), 400

    dataset_dir = Path("artifacts/data_ingestion/CT-KIDNEY-DATASET") / cat_normalized
    if not dataset_dir.exists():
        return jsonify({"error": f"Sample dataset directory not found at {dataset_dir}"}), 404

    sample_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
    if not sample_files:
        return jsonify({"error": f"No sample files found for category {category}"}), 404

    return send_file(sample_files[0], mimetype="image/jpeg")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    """Executes the training pipeline using the current Python environment."""
    try:
        logger.info("Initiating training pipeline from web request...")
        exit_code = os.system(f'"{sys.executable}" main.py')
        if exit_code == 0:
            return jsonify({"status": "success", "message": "Training completed successfully!"})
        else:
            return jsonify({"status": "error", "message": f"Training process returned code {exit_code}"}), 500
    except Exception as e:
        logger.exception(e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    """Predicts kidney condition from a base64 encoded image."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify([{"error": "Missing 'image' base64 string in JSON body"}]), 400

        image_str = data["image"]
        # Strip Data URL scheme header (e.g., 'data:image/jpeg;base64,') if present
        if "," in image_str:
            image_str = image_str.split(",", 1)[1]

        decodeImage(image_str, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        logger.exception(e)
        return jsonify([{"error": str(e)}]), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
