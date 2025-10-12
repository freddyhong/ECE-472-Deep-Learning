import tensorflow_datasets as tfds
import tensorflow as tf

def load_CIFAR10(batch_size=128, validation_size=5000, seed=0):
    ds_train = tfds.load("cifar10", split="train", shuffle_files=True, as_supervised=True)
    ds_test  = tfds.load("cifar10", split="test", as_supervised=True)

    def _preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return image, label
    
    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test  = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds_val = ds_train.take(validation_size)
    ds_train = ds_train.skip(validation_size)

    def augmentation(image, label):
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        return image, label
    
    ds_train = ds_train.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.shuffle(10 * batch_size, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test

def load_CIFAR100(batch_size=128, validation_size=5000, seed=0):
    ds_train = tfds.load("cifar100", split="train", shuffle_files=True, as_supervised=True)
    ds_test  = tfds.load("cifar100", split="test", as_supervised=True)

    def _preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return image, label
    
    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test  = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds_val = ds_train.take(validation_size)
    ds_train = ds_train.skip(validation_size)

    def augmentation(image, label):
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        return image, label
    
    ds_train = ds_train.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.shuffle(10 * batch_size, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test
