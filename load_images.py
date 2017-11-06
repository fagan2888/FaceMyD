import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Make a queue of file names including all the JPEG images files in the relative
# image directory.

filenames = []

for user in os.listdir("./Data/"):
    for filename in os.listdir("./Data/" + user):
        filenames.append("./Data/" + user + "/" + filename)    


filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(filenames))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)

# Start a new session to show example output.
with tf.Session() as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    #print(sess.run(tf.shape(image_tensor)))
    print(len(image_tensor))
    print(image_tensor[0].shape)
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
