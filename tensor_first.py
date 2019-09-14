import tensorflow as tf
tf.enable_eager_execution()


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)


for image, label in dataset:
    print(label)