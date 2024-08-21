import tensorflow as tf
from tensorflow import keras
from keras import layers, metrics
import numpy as np
import matplotlib as plt
import datetime

SIZE = 300
SIZE_IMAGE = (SIZE,SIZE)

# criando um classificador para as doenças com base nas pastas de treino
TRAINING_DIR = "detecção de plantas 15g/training"
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=SIZE_IMAGE,
                                                       class_mode='categorical')

# criando um validador para verificação da precisão das doenças
VALIDATION_DIR = "detecção de plantas 15g/Valid"
Validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

validation_generator = Validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size = SIZE_IMAGE,
                                                              class_mode='categorical')

#TEST_DIR = "/content/drive/MyDrive/images harvest/Test"
#test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#    rescale=1./255)

#test_generator = test_datagen.flow_from_directory(TEST_DIR,
#                                                  target_size= SIZE_IMAGE,
#                                                    class_mode='categorical')

batch_size = 16
epochs = 15
learning_rate = 0.001

# definindo a quantidade de filtros por camada
filters_layer_conv1 = 512
filters_layer_conv2 = 64
filters_layer_conv3 = 256
filters_layer_conv4 = 64
filters_layer_dense = 512
filters_layer_out = 2
# instanciando o modelo
model = tf.keras.models.Sequential(# construindo as camadas convolucionais
                                   [layers.Conv2D(filters_layer_conv1, (4, 4), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(3, 3),
                                    layers.Conv2D(filters_layer_conv2, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    layers.Conv2D(filters_layer_conv3, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    layers.Conv2D(filters_layer_conv4, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    layers.Conv2D(filters_layer_conv3, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    #Definindo a porcentagens de neuroniso que serão desligados após cada geração
                                    layers.Dropout(0.5),
                                    #compactação das camadas
                                    layers.Flatten(),
                                    # construindo a camada densa de entrada
                                    layers.Dense(filters_layer_dense, activation='relu'),
                                    #Construindo a camada densa de saída com a quantidade de filtro correspondendo a quantidade de classes
                                    layers.Dense(filters_layer_out, activation="softmax")])
#imprime as configurações das camadas, os parametros de saída e a quantidade de parametros
model.summary()
#compila as camadas do modelo e faz a otimização
history = model.compile(loss=tf.keras.losses.BinaryCrossentropy() ,
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=["accuracy"])
#Divide os dados em um conjunto de treinamento e um conjunto de validação, e usa o conjunto de validação para medir o progresso durante o treino.
model.fit(train_generator, batch_size= batch_size,epochs = epochs, validation_data=validation_generator)

#salva os dados do modelo
hora_atual = datetime.datetime.now()
data_hora = hora_atual.strftime("%Y-%m-%d %H:%M")
model.save('Models/Model_Plants/Plants_tf_version{} {}bs {}epochs {}.keras'.format(tf.__version__,batch_size,epochs,data_hora))