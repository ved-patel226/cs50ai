import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tqdm import tqdm
from termcolor import cprint
from alive_progress import alive_it
from time import sleep


class faceRecognition:
    def __init__(self):
        self.batch_size = 32
        self.img_height = 188
        self.img_width = 188

        # Use the parent directory of your label folders
        self.train_dir = r"images"
        self.train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )

        self.val_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
        )

    def model(self):
        self.base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3),
        )
        self.x = self.base_model.output
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dense(512, activation="relu")(self.x)
        self.predictions = Dense(
            len(self.train_generator.class_indices), activation="softmax"
        )(self.x)

        self.model = Model(inputs=self.base_model.input, outputs=self.predictions)
        for layer in alive_it(self.base_model.layers):
            layer.trainable = False
            sleep(0.1)

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self):
        self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=5,
        )

        self.model.save("model.h5")

    def load(self):
        self.model = tf.keras.models.load_model("model.h5")


face = faceRecognition()
face.model()
face.train()
