import tensorflow as tf
import numpy as np

def generator(project_shape, filters_list, strides_list, name="generator",setting='convergence'):
    model = tf.keras.Sequential(name=name)
    
    upscale      = np.prod(strides_list)
    img_height   = project_shape[0]*upscale
    img_width    = project_shape[1]*upscale
    img_channels = filters_list[-1]

    if(setting=='convergence' or setting=='divergence'):
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

    elif(setting=='mode_collapse'):
        for _ in range(5):
            model.add(tf.keras.layers.Dense(
            units=1024,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU(0.3))
        
        model.add(tf.keras.layers.Dense(
            units=img_height*img_width*img_channels,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))
        model.add(tf.keras.layers.Reshape(target_shape=[img_height,img_width,img_channels]))
    return model


def discriminator(filters_list, strides_list, name="discriminator",setting='convergence'):
    model = tf.keras.Sequential(name=name)
    if(setting=='convergence'  or setting=='divergence'):
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
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))

        return model
    
    elif(setting=='mode_collapse'):
        model.add(tf.keras.layers.Flatten())
        for _ in range(4):
            model.add(tf.keras.layers.Dense(
            units=1024,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU(0.3))

        model.add(tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))
        return model


class WGAN_GP(object):
    def __init__(
        self,
        project_shape,
        gen_filters_list,
        gen_strides_list,
        disc_filters_list,
        disc_strides_list,
        setting='convergence',
        gpl = 10
    ):
        if(setting=='mode_collapse' or setting =='divergence'):
            self.gpl = -0.001
        else:
            self.gpl = gpl

        self.project_shape = project_shape
        self.gen_filters_list = gen_filters_list
        self.gen_strides_list = gen_strides_list
        self.disc_filters_list = disc_filters_list
        self.disc_strides_list = disc_strides_list

        self.generator = generator(self.project_shape, self.gen_filters_list, self.gen_strides_list,setting=setting)
        self.discriminator = discriminator(self.disc_filters_list, self.disc_strides_list,setting=setting)
    
    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1, 1], minval=0., maxval=1.)
        interpolated = alpha * real + (1 - alpha) * fake
    
        with tf.GradientTape() as tape_p:
            tape_p.watch(interpolated)
            logit = self.discriminator(interpolated)
        
        grad = tape_p.gradient(logit, interpolated)
        grad_norm = tf.norm(tf.reshape(grad, (real.shape[0], -1)), axis=1)

        return self.gpl * tf.reduce_mean(tf.square(grad_norm - 1.))

    def generator_loss(self, z):
        x = self.generator(z, training=True)
        fake_score = self.discriminator(x, training=True)
        loss = -tf.reduce_mean(fake_score)
        return loss
    
    def discriminator_loss(self, x, z):
        x_fake = self.generator(z, training=True)
        true_score = self.discriminator(x, training=True)
        fake_score = self.discriminator(x_fake, training=True)
        gp = self.gradient_penalty(x,x_fake)
        loss = -tf.reduce_mean(true_score) + tf.reduce_mean(fake_score) + gp
        return loss