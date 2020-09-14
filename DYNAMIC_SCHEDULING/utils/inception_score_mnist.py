
from tensorflow_gan.examples.mnist import util as eval_util
import tensorflow as tf

def get_inception_score(images, splits=10):
    images = tf.convert_to_tensor(images)
    generated_mnist_score = eval_util.mnist_score(images)
    print(generated_mnist_score)
    return [generated_mnist_score]


def get_fid(real_images,generated_images, splits=10):
    real_images = tf.convert_to_tensor(real_images)
    generated_images = tf.convert_to_tensor(generated_images)
    generated_mnist_score = eval_util. mnist_frechet_distance(real_images,generated_images)
    print(generated_mnist_score)
    return [generated_mnist_score]