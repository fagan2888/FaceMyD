import tensorflow as tf
import glob, os, sys
from tensorflow.python.framework import ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filelist = glob.glob("./Data/Gudbrand/*.jpg")

def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, record_string = reader.read(filename_queue)
    example = tf.image.decode_jpeg(record_string)
    label = tf.reshape(key, [1], name=None)
    return example, label

def input_pipeline(filenames, batch_size, num_epochs = None):
	filename_queue = tf.train.string_input_producer(
	filelist, num_epochs=num_epochs, shuffle = True)

	example, label = read_my_file_format(filename_queue)
	label = (tf.string_split(label, "/").values)[2]

	#example.set_shape([720, 1080, 3])
	#label.set_shape([])

	# example_batch, label_batch = tf.train.shuffle_batch(
 #      [example, label],
 #      batch_size=batch_size,
 #      num_threads=4,
 #      capacity=50000,
 #      min_after_dequeue=10000)

	return example, label, var

example_batch, label_batch, size = input_pipeline(filelist, 1)


with tf.Session() as sess:
   # Required to get the filename matching to run.
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

  	# Coordinate the loading of image files.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	x = sess.run(example_batch)
	print(x.shape)
	y = sess.run(label_batch)
	print(y.decode("utf-8"))
	s = sess.run(size)
	print(s)





 # Finish off the filename queue coordinator.
	coord.request_stop()
	coord.join(threads)


