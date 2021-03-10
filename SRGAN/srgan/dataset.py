"""
Tensorflow data pipeline is very very powerful but you need to know how to utilize it
good learning
"""

##### Importing the neccesary imports
import tensorflow as tf
from srgan.config import config
import os

#### Setting the memory growth so all the GPU will not be allocated.
gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

@tf.function
def process_path(file_path):
  image = tf.io.read_file(file_path)
  image = tf.image.decode_png(image, channels=3)
  resize = tf.image.resize(image, size=(64, 64))
  image = tf.cast(image , dtype = tf.float32)
  resize = tf.cast(resize , dtype = tf.float32)
  return image, resize

@tf.function
def augment(image, resized):

  if tf.random.uniform((), minval=0, maxval=1) < 0.1:
    image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    resized = tf.tile(tf.image.rgb_to_grayscale(resized), [1, 1, 3])

  image = tf.image.random_brightness(image, max_delta=0.1)
  resized = tf.image.random_brightness(resized, max_delta=0.1)

  # a left upside down flipped is still a dog ;)
  image = tf.image.random_flip_left_right(image)  # 50%
  resized = tf.image.random_flip_left_right(resized)
  # image = tf.image.random_flip_up_down(image) #%50%

  return image, resized

def get_dataset(file_path):
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  BATCH_SIZE = config.TRAIN.batch_size

  dataset = tf.data.Dataset.list_files(file_path)
  dataset = dataset.map(process_path)
  dataset = dataset.map(augment , num_parallel_calls = 1)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(AUTOTUNE)

  return dataset

if __name__ == "__main__":
  dataset = get_dataset(config.TRAIN.img_dir)


  data = next(iter(dataset))
  print(data[0].shape)