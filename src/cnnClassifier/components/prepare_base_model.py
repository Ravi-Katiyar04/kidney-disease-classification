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
        inputs = tf.keras.layers.Input(
            shape=self.config.params_image_size,
            name="image_input"
        )
        # Rescaling layer normalizes [0, 255] to [-1, 1] for ResNet50V2
        x = tf.keras.layers.Rescaling(
            scale=1.0 / 127.5,
            offset=-1.0,
            name="resnet_rescaling"
        )(inputs)

        base_model = tf.keras.applications.ResNet50V2(
            input_tensor=x,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.model = base_model

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
                # Critical for transfer learning: preserve pretrained BN population statistics
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
        else:
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        # Classification head
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="gap"
        )(model.output)

        x = tf.keras.layers.Dense(
            units=256,
            activation="relu",
            name="dense_256"
        )(x)

        x = tf.keras.layers.Dropout(
            rate=0.3,
            name="dropout"
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
        freeze_till = getattr(self.config, "params_freeze_till", 35)
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=freeze_till,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(str(path))
