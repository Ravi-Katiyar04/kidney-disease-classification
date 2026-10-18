import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetV2B0(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(
        model,
        classes,
        freeze_all,
        freeze_till,
        learning_rate
    ):
        # Freeze pretrained layers as configured
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
            for layer in model.layers[-freeze_till:]:
                layer.trainable = True

        # Classification head
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        )(model.output)

        x = tf.keras.layers.BatchNormalization(
            name="batch_norm"
        )(x)

        x = tf.keras.layers.Dropout(
            rate=0.3,
            name="dropout_1"
        )(x)

        x = tf.keras.layers.Dense(
            units=256,
            activation="relu",
            name="dense_relu"
        )(x)

        x = tf.keras.layers.Dropout(
            rate=0.2,
            name="dropout_2"
        )(x)

        # Four classes: Normal, Tumor, Stone, Cyst
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            name="predictions"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=20,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(str(path))
