import pandas as pd
import os

# supresses all text from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tqdm import tqdm
from termcolor import cprint
from alive_progress import alive_it
from time import sleep


class butterflyClassification:
    def __init__(self):

        self.batch_size = 32
        self.img_height = 224
        self.img_width = 224

        self.test_csv = "Testing_set.csv"
        self.train_csv = "Training_set.csv"

        self.train_dir = "train"
        self.test_dir = "test"

        self.train_df = pd.read_csv(self.train_csv)
        self.test_df = pd.read_csv(self.test_csv)

        self.train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
        self.test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.train_generator = self.train_datagen.flow_from_dataframe(
            self.train_df,
            directory=self.train_dir,
            x_col="filename",
            y_col="label",
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )

        self.validation_generator = self.train_datagen.flow_from_dataframe(
            self.train_df,
            directory=self.train_dir,
            x_col="filename",
            y_col="label",
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
        )

        self.test_generator = self.test_datagen.flow_from_dataframe(
            self.test_df,
            directory=self.test_dir,
            x_col="filename",
            y_col=None,
            target_size=(224, 224),
            batch_size=32,
            class_mode=None,
            shuffle=False,
        )

    def model(self):
        self.base_model = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        self.x = self.base_model.output
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dense(512, activation="relu")(self.x)
        self.predictions = Dense(75, activation="softmax")(self.x)

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
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=10,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.n
            // self.validation_generator.batch_size,
        )

        self.model.save("butterfly.h5")

    def load(self):
        self.model = tf.keras.models.load_model("butterfly.h5")

    def evaluate(self):
        val_loss, val_accuracy = self.model.evaluate(self.validation_generator)
        cprint(f"Validation accuracy: {val_accuracy * 100:.2f}%", "green")

    def predict(self, img):
        image_path = img
        image = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)

        image_array = image_array / 255.0

        predictions = self.model.predict(image_array)
        predicted_class = tf.argmax(predictions[0]).numpy()

        reversed = dict(
            [(value, key) for key, value in self.train_generator.class_indices.items()]
        )
        predicted_class = reversed[predicted_class]

        cprint(f"Predicted butterfly: {predicted_class}", "green", attrs=["bold"])
        cprint(
            f"Confidence: {predictions[0][tf.argmax(predictions[0]).numpy()]*100:.2f}%",
            "green",
            attrs=["bold"],
        )
        cprint(f"File: {img}", "green", attrs=["bold"])


butteryfly = butterflyClassification()
butteryfly.load()
butteryfly.predict("mangrove_skipper.jpg")
butteryfly.predict("AFRICAN GIANT SWALLOWTAIL.jpg")
