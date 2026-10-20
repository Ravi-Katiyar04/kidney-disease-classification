import os
import sys
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from urllib.parse import urlparse
from dotenv import load_dotenv

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger

load_dotenv()


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.valid_generator = None
        self.model = None
        self.score = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bicubic",
            class_mode="categorical"
        )

        valid_datagenerator = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            seed=42,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)

        self._valid_generator()

        self.score = self.model.evaluate(
            self.valid_generator,
            verbose=1
        )

        self.save_score()

    def save_score(self):
        metric_names = getattr(self.model, "metrics_names", ["loss", "accuracy"])
        scores = {}
        for idx, (name, val) in enumerate(zip(metric_names, self.score)):
            clean_name = "accuracy" if name == "compile_metrics" or idx == 1 else name
            scores[clean_name] = float(val)

        save_json(
            path=Path("scores.json"),
            data=scores
        )

    def log_into_mlflow(self):
        load_dotenv()

        # Fix Windows console UTF-8 encoding issue for MLflow log runner emojis
        if sys.platform == "win32":
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")

        # Configure the DagsHub MLflow tracking server
        tracking_uri = self.config.mlflow_uri
        mlflow.set_tracking_uri(tracking_uri)

        # Ensure credentials exist when using remote tracking
        if urlparse(tracking_uri).scheme != "file":
            username = os.environ.get("MLFLOW_TRACKING_USERNAME")
            password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
            if not username or not password:
                raise ValueError(
                    "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be set in "
                    "your environment or .env file to log runs to remote MLflow server."
                )

        # Use a valid experiment name instead of a stale experiment ID
        experiment_name = "Kidney-Disease-Classification"

        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        if self.model is None or self.score is None:
            raise RuntimeError(
             "Run evaluation() before log_into_mlflow()."
            )

        # Get sample validation images
        sample_images, _ = next(iter(self.valid_generator))
        sample_images = sample_images[:5]

        # Infer the model signature
        sample_predictions = self.model.predict(
            sample_images,
            verbose=0
        )

        signature = infer_signature(
            sample_images,
            sample_predictions
        )

        # Start the run in the explicitly selected experiment
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(self.config.all_params)

            metric_names = getattr(self.model, "metrics_names", ["loss", "accuracy"])
            metrics_dict = {}
            for idx, (name, val) in enumerate(zip(metric_names, self.score)):
                clean_name = "accuracy" if name == "compile_metrics" or idx == 1 else name
                metrics_dict[clean_name] = float(val)
            mlflow.log_metrics(metrics_dict)

            mlflow.keras.log_model(
                model=self.model,
                name="model",
                signature=signature,
                registered_model_name="VGG16Model"
            )