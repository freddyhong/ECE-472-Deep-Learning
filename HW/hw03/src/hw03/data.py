import tensorflow_datasets as tfds
import tensorflow as tf

def load_mnist(batch_size=128, validation_size=10000, seed=0):
    ds_train = tfds.load("mnist", split="train", shuffle_files=True, as_supervised=True)
    ds_test  = tfds.load("mnist", split="test", as_supervised=True)

    def _preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return image, label

    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test  = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds_val = ds_train.take(validation_size)
    ds_train = ds_train.skip(validation_size)
    
    ds_train = ds_train.shuffle(10 * batch_size, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test
