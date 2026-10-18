import os
import urllib.request as request
import tensorflow as tf
import time
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        # Note: EfficientNetV2 includes internal rescaling & normalization [0, 255]
        datagenerator_kwargs = dict(
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:2]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

        valid_datagenerator = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data_path),
            subset="validation",
            shuffle=False,
            seed=42,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = (
                tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=20,
                    horizontal_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    **datagenerator_kwargs
                )
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data_path),
            subset="training",
            shuffle=True,
            seed=42,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(str(path))

    def train(self):
        # Compute balanced class weights to prevent majority-class collapse
        labels = self.train_generator.classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("Calculated class weights:", class_weight_dict)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epoch,
            validation_data=self.valid_generator,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
