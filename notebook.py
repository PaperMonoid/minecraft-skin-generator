from keras import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Conv2DTranspose
import tensorflow as tf


# Define model
model = Sequential()

# Uncomment if needed later
# model.add(BatchNormalization())
#model.add(MaxPooling2D(2))
#model.add(Conv2D(2, 32))
#model.add(UpSampling2D(2))
#model.add(Conv2D(32, 2))
#model.add(UpSampling2D(2))
#model.add(Conv2D(4, 2))
#model.add(Activation("relu"))


model.add(Input(shape=(64, 64, 3)))

model.add(Conv2D(64, 3, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(128, 3, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(256, 3, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2DTranspose(256, 3, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2DTranspose(64, 3, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2DTranspose(128, 3, strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2DTranspose(32, 3, strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(4, 2))
model.add(Activation("tanh"))

# Compile model
#optimizer = "rmsprop"
optimizer = "adam"

#loss = "categorical_crossentropy"
loss = "mse"
#loss = "mae"
#loss = "binary_crossentropy"
#loss = "sparse_categorical_crossentropy"
metrics = [
    "accuracy"
]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

INPUT_WIDTH = model.input_shape[1]
INPUT_HEIGHT = model.input_shape[2]
INPUT_DIMENTIONS = model.input_shape[3]

OUTPUT_WIDTH = model.output_shape[1]
OUTPUT_HEIGHT = model.output_shape[2]
OUTPUT_DIMENTIONS = model.output_shape[3]

# Model summary
model.summary()






from google.colab import drive
drive.mount('/content/drive')





import random
import os
from PIL import Image
import numpy as np


INPUT_DIRECTORY = "drive/My Drive/Colab Notebooks/MINECRAFT_MODEL/dataset/input"
OUTPUT_DIRECTORY = "drive/My Drive/Colab Notebooks/MINECRAFT_MODEL/dataset/output"


def binary_search(values, key):
    lower = 0
    upper = len(values)
    while lower + 1 < upper:
        half = int((lower + upper) / 2)
        if key == values[half]:
            return half
        elif key > values[half]:
            lower = half
        elif key < values[half]:
            upper = half
    return -lower


def load_data_index():
  input_files = []
  input_paths = []
  for filename in os.listdir(INPUT_DIRECTORY):
      path = os.path.join(INPUT_DIRECTORY, filename)
      if os.path.isfile(path):
        input_files.append(filename)
        input_paths.append(path)
  
  output_files = []
  output_paths = []
  for filename in os.listdir(OUTPUT_DIRECTORY):
      path = os.path.join(OUTPUT_DIRECTORY, filename)
      if os.path.isfile(path):
        output_files.append(filename)
        output_paths.append(path)
  
  input_files.sort()
  input_paths.sort()

  output_files.sort()
  output_paths.sort()

  data_index = []
  for i, input_file in enumerate(input_files):
    o = binary_search(output_files, input_file)
    if o > -1:
      data_index.append(
          [input_paths[i], output_paths[o]]
      )

  return data_index


def load_data(metadata):
  [input_path, output_path] = metadata

  input_image = Image.open(input_path)
  #input_image = input_image.resize((32, 32))
  output_image = Image.open(output_path)

  return (np.asarray(input_image), 
          np.asarray(output_image.convert(mode="RGBA")))


data_index = load_data_index()













from datetime import datetime

# train model
BATCH_SIZE = 3
EPOCHS = 5



processed = 0
total = len(data_index)


random.shuffle(data_index)
data_index_iterator = iter(data_index)
try:
    metadata = next(data_index_iterator)
except StopIteration:
    metadata = None

while metadata:

    batch_size = BATCH_SIZE
    x_batch = []
    y_batch = []

    for _ in range(BATCH_SIZE):
        try:
            metadata = next(data_index_iterator)
            [x_data, y_data] = load_data(metadata)
            x_batch.append(x_data)
            y_batch.append(y_data)
        except StopIteration:
            metadata = None

    batch_size = len(x_batch)
    processed += batch_size
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)

    history = model.fit(
        x_batch, y_batch, batch_size=batch_size, epochs=EPOCHS, verbose=0
    )
    loss = history.history["loss"][EPOCHS - 1]
    accuracy = history.history["accuracy"][EPOCHS - 1]
    timestamp = datetime.now()
    progress = processed / total * 100

    print(
        f"loss: {loss:.2f}, accuracy: {accuracy:.2f}, timestamp: {timestamp}, progress {progress:.2f}%"
    )







from IPython.display import display
#import tensorflow as tf
from PIL import Image

def display_array(np_array):
    #img = tf.keras.preprocessing.image.array_to_img(np_array)
    if np_array.shape[2] == 4:
      img = Image.fromarray(np_array, mode='RGBA')
    else:
      img = Image.fromarray(np_array)
    display(img)


[x_data, y_data] = load_data(data_index[0])

print("INPUT: ")
display_array(x_data)

print("CORRECT: ")
display_array(y_data)

pred = model.predict(np.array([x_data]))
print("PREDICTION: ")
display_array(pred[0])