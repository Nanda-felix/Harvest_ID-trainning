import tensorflow as tf
from keras import layers
import numpy as np
from keras import load_img, img_to_array
import datetime
import os

# Definição do Modelo
SIZE = 256
filters_layer_conv1 = 64
filters_layer_conv2 = 64
filters_layer_conv3 = 128
filters_layer_conv4 = 128
filters_layer_dense = 512
filters_layer_out = 2

model = tf.keras.models.Sequential([
    layers.Conv2D(filters_layer_conv1, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(filters_layer_conv2, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(filters_layer_conv3, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(filters_layer_conv4, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(filters_layer_dense, activation='relu'),
    layers.Dense(filters_layer_out, activation="softmax")
])

model.summary()

# Função para Processamento de Imagens
def process_image(img_path, img_width, img_height):
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Configurações de Treinamento
TRAINING_DIR = "D:/Harvest/Harvest-ID_training/dataset/trainning"  # Altere para o caminho correto
VALIDATION_DIR = "D:/Harvest/Harvest-ID_training/dataset/validation"  # Altere para o caminho correto
batch_size = 256
epochs = 10
learning_rate = 0.0001

# Geradores de Dados
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(256, 256), class_mode='categorical')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(256, 256), class_mode='categorical')

# Compilação e Treinamento do Modelo
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
model.fit(train_generator, batch_size=batch_size, epochs=epochs, validation_data=validation_generator, verbose=1, validation_steps=3)

# Salvando o Modelo
hora_atual = datetime.datetime.now()
data_hora = hora_atual.strftime("%Y-%m-%d %H:%M")
model.save('/path/to/saved_model/{}bs {}epochs {}.keras'.format(batch_size, epochs, data_hora))

