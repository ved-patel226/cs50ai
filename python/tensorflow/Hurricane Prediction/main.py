import pandas as pd
import numpy as np
from datetime import datetime
from termcolor import cprint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

# quit because of lack of data


class hurricane:
    def __init__(self):
        self.dataX = []
        self.dataY = []

    def __lat_lon_to_float(self, v):
        if (v[-1] == "S") or (v[-1] == "W"):
            multiplier = -1
        else:
            multiplier = 1
        return float(v[:-1]) * multiplier

    def load_data(self):
        with open(r"data\fatalities_c.csv", "r") as f:
            for line in f.readlines():
                line = line.split(",")

                if line[0] == "fatality_id":
                    continue

                with open("data\details_c.csv", "r") as f:
                    for linet in f.readlines():
                        linet = linet.split(",")
                        if linet[0] == "last_date_modified":
                            continue
                        if line[0] in linet:
                            self.dataY.append(int(linet[20].strip('"')))
                            try:
                                self.dataX.append(
                                    [
                                        int(line[0].strip('"')),
                                        int(line[1].strip('"')),
                                        ord(line[2].strip('"')),
                                        int(line[4].strip('"')),
                                        int(
                                            "".join(
                                                [str(ord(char)) for char in line[6]]
                                            )
                                        ),
                                    ]
                                )
                                self.dataY.append(int(linet[20].strip('"')))
                            except:
                                continue
        print(self.dataX)

    def model(self):
        trainX, testX = self.__split_list(self.dataX)
        trainY, testY = self.__split_list(self.dataY)

        print(len(trainX))
        print(len(trainY))

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["accuracy"],
        )

        trainX = [trainX[0]]
        trainY = trainY[0]

        trainX = np.array(trainX)
        trainY = np.array(trainY)

        model.fit(trainX, trainY, epochs=3)

        # model.save("modelv0.keras")

    def __split_list(self, lst):
        half = len(lst) // 2
        return lst[:half], lst[half:]


h = hurricane()
h.load_data()
