import tensorflow as tf
import glob, os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_image(filename):

	contents = tf.read_file(filename)
	image = tf.image.decode_jpeg(contents)
	image = tf.image.convert_image_dtype(image, tf.float32)

	return image

def repeat_images(images, N):
	return tf.tile(images,[N,1,1,1])

def get_windows(images):

	offsets = tf.random_uniform([100,2], minval=-200, maxval=200, dtype=tf.int32)
	offsets = tf.cast(offsets, tf.float32)

	windows = tf.image.extract_glimpse(images, [128, 128], offsets, normalized=False)

	return windows

def process_windows(windows):
	return tf.map_fn(tf.image.per_image_standardization, windows)

def encode_jpegs(im):
	im = tf.image.convert_image_dtype(im, tf.uint8)
	return tf.image.encode_jpeg(im)


if __name__ == "__main__":

	N = 10

	user = sys.argv[1]
	filenames = glob.glob("./Data/Raw/{}/*.jpg".format(user))
	filenames = tf.constant(filenames)

	images = tf.map_fn(load_image, filenames, dtype=tf.float32)

	images_repeated = repeat_images(images, N)

	windows = get_windows(images_repeated)

	processed = process_windows(windows)

	encoded = tf.map_fn(encode_jpegs, windows, dtype=tf.string)

	with tf.Session() as sess:

	    strings = sess.run(encoded)

	    for i in range(strings.shape[0]):
	    	with open("./Data/Processed/{}/{}{}_processed.jpg".format(user, user, i), "wb+") as out:
	    		out.write(strings[i])






