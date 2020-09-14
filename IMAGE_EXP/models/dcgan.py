import tensorflow as tf
import numpy as np

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def _discriminator_loss(real_output, fake_output):
    real_loss  = tf.reduce_mean(tf.math.log(real_output+1e-4))
    fake_loss  = tf.reduce_mean(tf.math.log(1 - fake_output + 1e-4))
    total_loss = -(real_loss + fake_loss)/2

    return total_loss

def _generator_loss(fake_output):
    return tf.reduce_mean(tf.math.log(1 - fake_output + 1e-4))


def generator(project_shape, filters_list, strides_list, name="generator",setting='convergence'):
    model = tf.keras.Sequential(name=name)
    
    upscale      = np.prod(strides_list)
    img_height   = project_shape[0]*upscale
    img_width    = project_shape[1]*upscale
    img_channels = filters_list[-1]

    model.add(tf.keras.layers.Dense(
        units=np.prod(project_shape),
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape(target_shape=project_shape))
    for filters, strides in zip(filters_list[:-1], strides_list[:-1]):
        model.add(tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=[5, 5],
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2DTranspose(
        filters=filters_list[-1],
        kernel_size=[5, 5],
        strides=strides_list[-1],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
    ))
    return model


def discriminator(filters_list, strides_list, name="discriminator",setting='convergence'):
    model = tf.keras.Sequential(name=name)
    
    for filters, strides in zip(filters_list, strides_list):
        model.add(tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[5, 5],
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=1,
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
    ))

    return model


class DCGAN(object):
    def __init__(
        self,
        project_shape,
        gen_filters_list,
        gen_strides_list,
        disc_filters_list,
        disc_strides_list,
        setting='convergence'
    ):
        self.project_shape = project_shape
        self.gen_filters_list = gen_filters_list
        self.gen_strides_list = gen_strides_list
        self.disc_filters_list = disc_filters_list
        self.disc_strides_list = disc_strides_list

        self.generator = generator(self.project_shape, self.gen_filters_list, self.gen_strides_list,setting=setting)
        self.discriminator = discriminator(self.disc_filters_list, self.disc_strides_list,setting=setting)
    
    def generator_loss(self, z):
        x = self.generator(z, training=True) 
        fake_score = self.discriminator(x, training=True)
        loss = _generator_loss(fake_score)

        return loss
    
    def discriminator_loss(self, x, z):
        x_fake = self.generator(z, training=True)
        true_score = self.discriminator(x, training=True)
        fake_score = self.discriminator(x_fake, training=True)

        loss = _discriminator_loss(true_score,fake_score)
        
        return loss