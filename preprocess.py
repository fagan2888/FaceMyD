import tensorflow as tf
import numpy as np
import sys


"""
Takes a user id (name) as input, and returns a (N, m, n, 3) Tensor 
"""

SCALE_SIZE = 100 
NUM_CLASSES = 2
N = 10

def read_jpg(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_bmp(image_string)
    image = tf.image.resize_images(image_decoded, [SCALE_SIZE, SCALE_SIZE])
    one_hot = tf.one_hot(label, NUM_CLASSES)
    return image, one_hot



if __name__ == "__main__":
	user = sys.argv[1]

	user_images = tf.Tensor(	
	dataset1 = tf.data.Dataset.from_tensor_slices(images)
	images = []
	for i in range(N):
		
		filename = "./Data/" + user + "/" + user + str(i) + ".bmp"
		im, oh = read_jpg(filename, 1)
		print(im)

