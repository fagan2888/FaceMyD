import tensorflow as tf
import numpy as np


SCALE_SIZE = 100 
NUM_CLASSES = 2

def read_bmp(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_bmp(image_string)
    image = tf.image.resize_images(image_decoded, [SCALE_SIZE, SCALE_SIZE])
    one_hot = tf.one_hot(label, NUM_CLASSES)
    return image, one_hot



if __name__ == "__main__":
	filename = "./Data/Gudbrand/Gudbrand1.bmp"
	im, oh = read_bmp(filename, 1)
	print(oh)

