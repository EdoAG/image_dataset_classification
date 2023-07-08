import warnings

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

warnings.filterwarnings('ignore')


class KerasClassification:
    def __init__(self):
        self.Train = None
        self.Test = None
        self.Validation = None
        self.Class_Names = None

    def get_datasets(self,train_path,test_path,valid_path):
        """
        Importa tutti i dati con batch da 32 immagini
        :return:
        """
        self.Train = tf.keras.utils.image_dataset_from_directory(train_path, label_mode='int',
                                                                 batch_size=32, image_size=(180, 180), seed=42)
        self.Test = tf.keras.utils.image_dataset_from_directory(test_path, label_mode='int',
                                                                batch_size=32, image_size=(180, 180), seed=42)
        self.Validation = tf.keras.utils.image_dataset_from_directory(valid_path, label_mode='int',
                                                                      batch_size=32, image_size=(180, 180), seed=42)
        self.Class_Names = self.Train.class_names

    def model(self):
        """
        modello: 15 layers , 16 epoche
        :return:
        """
        AUTOTUNE = tf.data.AUTOTUNE
        self.Train = self.Train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.Validation = self.Validation.cache().prefetch(buffer_size=AUTOTUNE)
        num_classes = len(self.Class_Names)
        model = tf.keras.Sequential([
            layers.Rescaling(1. / 255, input_shape=(180, 180, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPool2D(),
            layers.Dropout(.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        print(model.summary())
        history = model.fit(self.Train, validation_data=self.Validation, epochs=16)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print(acc, val_acc, loss, val_loss)
        epochs_range = range(16)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Accuracy in training')
        plt.plot(epochs_range, val_acc, label='Accuracy in validation')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Perdita in training')
        plt.plot(epochs_range, val_loss, label='Perdita in validation')
        plt.legend(loc='upper right')
        plt.show()

        print(model.evaluate(self.Test))
