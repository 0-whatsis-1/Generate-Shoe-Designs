import os
from PIL import Image
import tensorflow as tf
import numpy as np
import scipy.misc

HEIGHT, WIDTH = 64, 64
CHANNEL = 3
BATCH_SIZE = 64


def data_preprocess():
    current_dir = os.getcwd()
    shoe_dir = os.path.join(current_dir, 'Shoe')
    images = []
    for each_img in os.listdir(shoe_dir):
        images.append(os.path.join(shoe_dir, each_img))
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([all_images])
    content = tf.read_file(images_queue[0])
    img = tf.image.decode_jpeg(content, channels = CHANNEL)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta = 0.1)
    img = tf.image.random_contrast(img, lower = 0.9, upper = 1.1)
    img = tf.image.resize_images(img, [HEIGHT, WIDTH])
    img.set_shape([HEIGHT, WIDTH, CHANNEL])
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    images_batch = tf.train.shuffle_batch([img], batch_size = BATCH_SIZE,
        num_threads = 2, capacity = 200 + 2* BATCH_SIZE, min_after_dequeue = 200)
    no_of_images = len(images)

    return images_batch, no_of_images


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  return (images+1.)/2.
