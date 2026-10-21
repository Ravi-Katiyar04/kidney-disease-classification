import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import tensorflow as tf

from cnnClassifier import logger

# Mapping of index to kidney CT scan diagnosis based on training dataset
CLASS_INDICES: Dict[int, str] = {
    0: "Cyst",
    1: "Normal",
    2: "Stone",
    3: "Tumor",
}


class PredictionPipeline:
    """Pipeline for kidney CT scan classification."""

    _cached_model = None
    _cached_model_path: Optional[str] = None

    def __init__(
        self,
        filename: Union[str, Path],
        model_path: Optional[Union[str, Path]] = None,
    ):
        self.filename = Path(filename)
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            default_path = Path("artifacts/training/model.h5")
            fallback_path = Path("model/model.h5")
            if default_path.exists():
                self.model_path = default_path
            elif fallback_path.exists():
                self.model_path = fallback_path
            else:
                self.model_path = default_path

    @classmethod
    def get_model(cls, model_path: Path):
        """Loads and caches the model to avoid reloading for every prediction."""
        resolved_path = str(model_path.resolve())
        if cls._cached_model is None or cls._cached_model_path != resolved_path:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. "
                    "Please run the training pipeline first (e.g. `python main.py` or `dvc repro`)."
                )
            logger.info(f"Loading model from {model_path}...")
            cls._cached_model = tf.keras.models.load_model(str(model_path))
            cls._cached_model_path = resolved_path
            logger.info("Model loaded successfully into memory.")
        return cls._cached_model

    def predict(self) -> List[Dict[str, Union[str, float, int]]]:
        """Predicts the kidney disease class for the specified image."""
        if not self.filename.exists():
            raise FileNotFoundError(f"Input image not found: {self.filename}")

        model = self.get_model(self.model_path)

        # Load and preprocess image using tf.keras.utils (compatible with Keras 3 / TensorFlow 2.16+)
        test_image = tf.keras.utils.load_img(
            str(self.filename),
            target_size=(224, 224),
            interpolation="bicubic",
        )
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Run inference
        probabilities = model.predict(test_image, verbose=0)
        predicted_idx = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities, axis=1)[0])
        prediction = CLASS_INDICES.get(predicted_idx, "Unknown")

        logger.info(
            f"Prediction: {prediction} (class {predicted_idx}) with confidence {confidence * 100:.2f}%"
        )

        all_probabilities = {
            CLASS_INDICES[i]: round(float(probabilities[0][i]), 5)
            for i in range(len(CLASS_INDICES))
        }

        return [
            {
                "image": prediction,
                "confidence": confidence,
                "class_idx": predicted_idx,
                "probabilities": all_probabilities,
            }
        ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        pipeline = PredictionPipeline(img_path)
        result = pipeline.predict()
        print(f"Result for {img_path}: {result}")
    else:
        print("Usage: python src/cnnClassifier/pipeline/prediction.py <path_to_image>")