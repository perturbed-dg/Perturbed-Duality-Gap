""" This module implement a gan task """

import os
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from dataio.dataset_mnist import Dataset_mnist
import utils
from utils import save_images
from utils.analyse_utils import plot_to_image
from utils.inception_score_mnist import get_inception_score,get_fid
from models.basic_model import Basic_model
from scipy import stats
import pandas as pd

logger = utils.get_logger()


project_shape = [7, 7, 256]
gen_filters_list =  [128, 64, 1]
gen_strides_list = [1, 2, 2]
disc_filters_list =  [64, 128]
disc_strides_list = [2, 2]

class Gan(Basic_model):
    def __init__(self, config, exp_name='debug', arch=None):
        super(Gan, self).__init__(config, exp_name)
        
        self.arch = arch
        if arch:
            logger.info('architecture:')
            for key in sorted(arch.keys()):
                logger.info('{}: {}'.format(key, arch[key]))

        self.reset()
        self._load_datasets()
        self._build_placeholder()
        self._build_graph()

        self.inception_score = []
        self.fid = []
        self.reward_baseline = None
        self.lambda_reg = 2.0
        self.dg_values = []
        # ema of metrics track over episodes
        n_steps = int(config.max_training_step / config.valid_frequency_task)

        self.metrics_track_baseline = -np.ones([n_steps])
        self.fixed_noise_128 = np.random.normal(size=(config.vis_count, config.dim_z))\
            .astype('float32')

    def _load_datasets(self):
        config = self.config
        self.train_dataset = Dataset_mnist()
        self.train_dataset.load_mnist(config.data_dir,
                                      tf.estimator.ModeKeys.TRAIN)

        self.valid_dataset = Dataset_mnist()
        self.valid_dataset.load_mnist(config.data_dir,tf.estimator.ModeKeys.EVAL)

    def reset(self):
        # ----Reset the model.----
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = 0
        self.ema_gen_cost = None
        self.ema_disc_cost_real = None
        self.ema_disc_cost_fake = None
        self.prst_gen_cost = None
        self.prst_disc_cost_real = None
        self.prst_disc_cost_fake = None
        self.mag_gen_grad  = None
        self.mag_disc_grad = None
        self.duality_gap   = 10
        self.dg_ema        = 10

        # to control when to terminate the episode
        self.endurance = 0
        # The bigger the performance is, the better. In this case, performance
        # is the inception score. Naming it as performance in order to be
        # compatible with other tasks.
        self.best_performance = np.inf
        self.collapse = False
        self.previous_action = -1
        self.same_action_count = 0
        self.dg_values = []

        lr = self.config.lr_task
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        
        self.dg_g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        self.dg_d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
      
        
    def _build_placeholder(self):
        dim_x = self.config.dim_x
        dim_z = self.config.dim_z
        # self.real_data = tf.keras.layers.Input(shape=(dim_x,))
        # self.noise     = tf.keras.layers.Input(shape=(dim_z,))
        self.is_training = True

    def _train_G(self):
        with tf.GradientTape() as g_tape:
            # fake_data = self.generator(self.noise)
            # disc_fake = self.discriminator(fake_data)

            # gen_cost = -tf.reduce_mean(disc_fake)

            gen_cost = self._loss_g()

        gen_grad = g_tape.gradient(gen_cost, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_grad,self.generator.trainable_variables))
        # self.fake_data = fake_data
        self.gen_cost = gen_cost
        self.gen_grad = [ tf.keras.layers.Flatten()(x) for x in gen_grad]
    
    def _dg_train_G(self):
        with tf.GradientTape() as g_tape:
            # fake_data = self.dg_generator(self.noise)
            # disc_fake = self.dg_discriminator(fake_data)

            # gen_cost = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=disc_fake, labels=tf.ones_like(disc_fake)
            #     )
            # )


            gen_cost = self._dg_loss_g()

        gen_grad = g_tape.gradient(gen_cost, self.dg_generator.trainable_variables)
        self.dg_g_optimizer.apply_gradients(zip(gen_grad,self.dg_generator.trainable_variables))
        
    def _train_D(self):
        with tf.GradientTape() as d_tape:
            # real_data = tf.cast(self.real_data, tf.float32)
            # fake_data = self.generator(self.noise)
            # disc_real = self.discriminator(real_data)
            # disc_fake = self.discriminator(fake_data)

            # disc_cost = -tf.reduce_mean(disc_real) + tf.reduce_mean(disc_fake)

            disc_cost = self._loss_d()
        disc_grad = d_tape.gradient(disc_cost, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_grad,self.discriminator.trainable_variables))
        
        # for w in self.discriminator.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -0.1, 0.1))

        self.disc_cost_fake = disc_cost
        self.disc_cost_real = disc_cost
        self.disc_grad = [tf.keras.layers.Flatten()(x) for x in disc_grad]
        self.disc_cost = disc_cost

    def kde(self,values, fig_size=(8, 8), bbox=[-2.5, 2.5, -2.5, 2.5], xlabel="", ylabel="", cmap='RdPu', show=False, save=None):
    
        fig, ax = plt.subplots(figsize=fig_size)
        
        kernel = stats.gaussian_kde(values)
        
        if(self.config.data_task=='grid'):
            bbox=[-3.5, 2.5, -3.5, 2.5]
        ax.axis(bbox)
        xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cfset = ax.contourf(xx, yy, f, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        if save is not None: plt.savefig(save,bbox_inches='tight')
        if show: plt.show()

        plt.close() 

    def _dg_train_D(self):
        with tf.GradientTape() as d_tape:
            # real_data = tf.cast(self.real_data, tf.float32)
            # fake_data = self.dg_generator(self.noise)
            # disc_real = self.dg_discriminator(real_data)
            # disc_fake = self.dg_discriminator(fake_data)

            # disc_cost_fake = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=disc_fake, labels=tf.zeros_like(disc_fake)
            #     )
            # )
            # disc_cost_real = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=disc_real, labels=tf.ones_like(disc_real)
            #     )
            # )
            # disc_cost = (disc_cost_fake + disc_cost_real) / 2.

            disc_cost = self._dg_loss_d()
        disc_grad = d_tape.gradient(disc_cost, self.dg_discriminator.trainable_variables)

        self.dg_d_optimizer.apply_gradients(zip(disc_grad,self.dg_discriminator.trainable_variables))
        
        # for w in self.discriminator.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -0.1, 0.1))

        
    def _build_graph(self):
        lr = self.config.lr_task
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        self.generator     = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator_init     = self.build_generator()
        self.discriminator_init = self.build_discriminator()

        self.generator.set_weights(self.generator_init.get_weights())
        self.discriminator.set_weights(self.discriminator_init.get_weights())
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        
        self.dg_generator     = self.build_generator()
        self.dg_discriminator = self.build_discriminator()

        self.dg_g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        self.dg_d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta1,beta_2=beta2)
        
        self.saver =  tf.train.Checkpoint
        
        self.models_to_save.append(('generator',self.generator,self.g_optimizer))
        self.models_to_save.append(('discriminator',self.discriminator,self.d_optimizer))
        
        self.models_initial.append((self.generator,self.generator_init))
        self.models_initial.append((self.discriminator,self.discriminator_init))
        
        self.update = [self._train_G, self._train_D]

    def build_generator(self):

        activation = self.config.activation
        dim_z = self.config.dim_z
        dim_x = self.config.dim_x
        h_dim = self.config.h_dim
        n_layers = self.config.n_layers

        if activation == 'relu':
            activation_fn =  tf.keras.layers.Activation(tf.nn.relu)
        elif activation == 'leakyRelu':
            activation_fn =  tf.keras.layers.Activation(tf.nn.leaky_relu)
        else:
            activation_fn =  tf.keras.layers.Activation(tf.nn.tanh)

        filters_list = gen_filters_list
        strides_list = gen_strides_list
        model = tf.keras.models.Sequential()
        # model.add( tf.keras.layers.Dense(h_dim, input_dim=dim_z))
        # model.add(activation_fn)
        
        # for _ in range(n_layers-1):
        #     model.add( tf.keras.layers.Dense(h_dim))
        #     model.add(activation_fn)
        
        # model.add( tf.keras.layers.Dense(dim_x))
        # model.summary()

        #----------------------------------------------------
        
        
        model.add(tf.keras.layers.Dense(
            units=np.prod(project_shape),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02), input_dim=dim_z
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
            # model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Conv2DTranspose(
            filters=filters_list[-1],
            kernel_size=[5, 5],
            strides=strides_list[-1],
            padding="same",
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))

        model.add(tf.keras.layers.Flatten())
        
        # --------------------------------------------------- 
        noise          = tf.keras.layers.Input(shape=(dim_z,))
        return tf.keras.Model(noise,model(noise))

    def build_discriminator(self):
        activation = self.config.activation
        
        dim_x = self.config.dim_x
        h_dim = self.config.h_dim
        n_layers = self.config.n_layers

        if activation == 'relu':
            activation_fn =  tf.keras.layers.Activation(tf.nn.relu)
        elif activation == 'leakyRelu':
            activation_fn =  tf.keras.layers.Activation(tf.nn.leaky_relu)
        else:
            activation_fn =  tf.keras.layers.Activation(tf.nn.tanh)


        filters_list = disc_filters_list
        strides_list = disc_strides_list
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Flatten())
        model.add( tf.keras.layers.Reshape((28,28,1)))
        # model.add( tf.keras.layers.Dense(h_dim, input_dim=dim_x))
        # model.add(activation_fn)
        
        # for _ in range(n_layers-1):
        #     model.add( tf.keras.layers.Dense(h_dim))
        #     model.add(activation_fn)
        
        # model.add(tf.keras.layers.Dense(1))

        # model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
#   

#--------------------------------------------------------------

        for filters, strides in zip(filters_list, strides_list):
                    model.add(tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=[5, 5],
                        strides=strides,
                        padding="same",
                        use_bias=False,
                        kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
                    ))
                    # model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=1,
            # activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02)
        ))

#-----------------------------------------------------------------

        # model.summary()
        input_  = tf.keras.layers.Input(shape=(dim_x,))
        return tf.keras.Model(input_,model(input_))
        # return model

    def _loss_d(self):
        x = self.real_data
        z = self.noise
        fake  = self.generator(z,training=True) 

        D_fake = self.discriminator(fake,training=True)
        D_real = self.discriminator(x,training=True)

        # print(x,fake)
        epsilon = tf.random.uniform(shape=tf.shape(x), minval=0., maxval=1.)
        interpolation = epsilon * x + (1 - epsilon) * fake
    
        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            D_interpolation = self.discriminator(interpolation,training=True)
        
        grads   = tape.gradient(D_interpolation,[interpolation])
        gp = (tf.norm(grads, axis=1) - 1) ** 2.0
        for gr in gp:
            tf.clip_by_value(gr,-1,1)
        loss =  tf.reduce_mean(D_fake - D_real + self.lambda_reg*gp)
        return loss
    
    def _dg_loss_d(self,score=False):
        x = self.real_data
        z = self.noise
        fake  = self.dg_generator(z,training=True) 

        D_fake = self.dg_discriminator(fake,training=True)
        D_real = self.dg_discriminator(x,training=True)

        epsilon = tf.random.uniform(shape=tf.shape(x), minval=0., maxval=1.)
        interpolation = epsilon * x + (1 - epsilon) * fake
    
        with tf.GradientTape() as tape:
            tape.watch(interpolation)
            D_interpolation = self.dg_discriminator(interpolation,training=True)
        
        grads   = tape.gradient(D_interpolation,[interpolation])
        gp = (tf.norm(grads, axis=1) - 1) ** 2.0
        for gr in gp:
            tf.clip_by_value(gr,-1,1)
        
        if(score):
            return tf.reduce_mean(D_fake - D_real)

        loss = tf.reduce_mean(D_fake - D_real + self.lambda_reg*gp)
        return loss
    
    def _loss_g(self):
        z = self.noise
        fake   = self.generator(z,training=True) 
        D_fake = self.discriminator(fake,training=True)
        loss = -tf.reduce_mean(D_fake)
        return loss

    def _dg_loss_g(self):
        z = self.noise
        fake   = self.dg_generator(z,training=True) 
        D_fake = self.dg_discriminator(fake,training=True)
        loss =  -tf.reduce_mean(D_fake)
        return loss

    def get_costs(self,z,x):
        real_data = tf.cast(x, tf.float32)
        fake_data = self.generator(z,training=False)
        disc_real = self.discriminator(real_data,training=False)
        disc_fake = self.discriminator(fake_data,training=False)

        # disc_cost_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=disc_fake, labels=tf.zeros_like(disc_fake)
        #     )
        # )
        # disc_cost_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=disc_real, labels=tf.ones_like(disc_real)
        #     )
        # )
        # disc_cost = (disc_cost_fake + disc_cost_real) / 2.
        # gen_cost = tf.reduce_mean(
        #                 tf.nn.sigmoid_cross_entropy_with_logits(
        #                     logits=disc_fake, labels=tf.ones_like(disc_fake)
        #                 )
        #             )
#-----------------------------------------------------------------------------------
        
        disc_cost_fake = tf.reduce_mean(disc_fake)
        disc_cost_real = -tf.reduce_mean(disc_real) 
        gen_cost = -tf.reduce_mean(disc_fake)

        return [gen_cost,disc_cost_fake,disc_cost_real]

    def train(self, save_model=True):
        # This function trains a GAN under fixed schedule.
        
        num_trials = 5
        for trial in range(num_trials):
            
            self.reset()
            # self.initialize_weights()
            self.init_model()
            config = self.config
            batch_size = config.batch_size
            dim_z = config.dim_z
            valid_frequency = config.valid_frequency_task
            print_frequency = config.print_frequency_task
            max_endurance = config.max_endurance_task
            endurance = 0

            best_dg     = np.inf
            dg_baseline = np.inf

            decay = config.metric_decay
            steps_per_iteration = config.disc_iters + config.gen_iters
            lr = config.lr_task
            
            for step in range(config.max_training_step):
                if step % steps_per_iteration < config.disc_iters:
                    # ----Update D network.----
                    data = self.train_dataset.next_batch(batch_size)
                    x = data['input']
                    z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)

                    self.noise     = z
                    self.real_data = tf.reshape(x,[x.shape[0],-1])
                    # print('\n Data Shape : ',self.real_data.shape)
                    self._train_D()
                else:
                    # ----Update G network.----
                    z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                    self.noise = z
                    self._train_G()

                if step % valid_frequency == 0:
                    logger.info('========TRIAL {} STEP{}========'.format(trial,step))
                    self.evaluate(step,[],trial)
                    logger.info(endurance)
                    duality_gap = self.get_duality_gap()
                    logger.info(duality_gap)
                    if dg_baseline < np.inf:
                        dg_baseline = dg_baseline * decay \
                            + duality_gap * (1 - decay)
                    else:
                        dg_baseline = duality_gap
                    logger.info('dg_baseline: {}'.format(dg_baseline))
                    self.dg_values.append(dg_baseline)
                    self.generate_images(step)
                    endurance += 1

                    
                    if dg_baseline < best_dg and step > 1000 :
                        best_dg = dg_baseline
                        endurance = 0
                        if save_model:
                            self.save_model(step)

                if step % print_frequency == 0:
                    data = self.train_dataset.next_batch(batch_size)
                    x = data['input']
                    z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                    
                    r = self.get_costs(z,x)
                    logger.info('gen_cost: {}'.format(r[0]))
                    logger.info('disc_cost fake: {}, real: {}'.format(r[1], r[2]))

                if endurance > max_endurance:
                    break
            logger.info('best_dg: {}'.format(best_dg))
    
    def get_grad_state(self):
        with tf.GradientTape() as g_tape,tf.GradientTape() as d_tape:
            real_data = tf.cast(self.real_data, tf.float32)
            fake_data = self.generator(self.noise)
            disc_real = self.discriminator(real_data)
            disc_fake = self.discriminator(fake_data)

            # disc_cost_fake = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=disc_fake, labels=tf.zeros_like(disc_fake)
            #     )
            # )
            # disc_cost_real = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=disc_real, labels=tf.ones_like(disc_real)
            #     )
            # )

            # disc_cost = (disc_cost_fake + disc_cost_real) / 2.
            # gen_cost = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=disc_fake, labels=tf.ones_like(disc_fake)
            #     )
            # )
            #-----------------------------------------------------------------------------
           

            disc_cost_fake = tf.reduce_mean(disc_fake)
            disc_cost_real = -tf.reduce_mean(disc_real) 
            disc_cost = (disc_cost_fake + disc_cost_real)/2
            gen_cost = -tf.reduce_mean(disc_fake)

        gen_grad  = d_tape.gradient(gen_cost, self.generator.trainable_variables)
        disc_grad = g_tape.gradient(disc_cost, self.discriminator.trainable_variables)
        
        return gen_grad, disc_grad, gen_cost, disc_cost_real, disc_cost_fake

    def response(self, action):
        """
         Given an action, return the new state, reward and whether dead

         Args:
             action: one hot encoding of actions

         Returns:
             state: shape = [dim_input_ctrl]
             reward: shape = [1]
             dead: boolean
        """
        # GANs only uses one dataset
        dataset = self.train_dataset

        config = self.config
        batch_size = config.batch_size
        dim_z = config.dim_z
        alpha = config.state_decay
        lr = config.lr_task
        a = np.argmax(np.array(action))

        # ----To detect collapse.----
        if a == self.previous_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        self.previous_action = a

        update_times = [config.gen_iters, config.disc_iters]

        for _ in range(update_times[a]):
            data = self.train_dataset.next_batch(batch_size)
            x = data['input']
            z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)

            self.noise     = z
            self.real_data = tf.reshape(x,[x.shape[0],-1])

            # print('\n Data Shape : ',self.real_data.shape)
            self.update[a]()
            gen_grad, disc_grad, gen_cost, disc_cost_real, disc_cost_fake = self.get_grad_state()

        self.mag_gen_grad  = self.get_grads_magnitude(gen_grad)
        self.mag_disc_grad = self.get_grads_magnitude(disc_grad)
        self.prst_gen_cost = gen_cost
        self.prst_disc_cost_real = disc_cost_real
        self.prst_disc_cost_fake = disc_cost_fake

        # ----Update state.----
        self.step_number += 1
        if self.ema_gen_cost is None:
            self.ema_gen_cost = gen_cost
            self.ema_disc_cost_real = disc_cost_real
            self.ema_disc_cost_fake = disc_cost_fake
        else:
            self.ema_gen_cost = self.ema_gen_cost * alpha\
                + gen_cost * (1 - alpha)
            self.ema_disc_cost_real = self.ema_disc_cost_real * alpha\
                + disc_cost_real * (1 - alpha)
            self.ema_disc_cost_fake = self.ema_disc_cost_fake * alpha\
                + disc_cost_fake * (1 - alpha)

        self.extra_info = {'gen_cost': self.ema_gen_cost,
                           'disc_cost_real': self.ema_disc_cost_real,
                           'disc_cost_fake': self.ema_disc_cost_fake}


        reward = 0
        # if(self.config.step_reward==True):
        #     reward = self.get_step_reward(self.step_number)
        
        tf.summary.scalar('Step Reward',reward,step=self.step_number)
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()

        # tf.summary.scalar('G Loss',self.gen_cost,step=self.step_number)
        # tf.summary.scalar('D Loss Real',self.gen_cost,step=self.step_number)
        # tf.summary.scalar('D Loss Fake',self.gen_cost,step=self.step_number)
        
        return state, reward, dead

    def update_duality_gap(self, score):
        self.best_performance = score

    def get_state(self):
        if self.step_number == 0:
            state = [0] * self.config.dim_input_ctrl
        else:
            state = [
                     math.log((self.mag_disc_grad / (1e-4+self.mag_gen_grad))+1e-4),
                     self.ema_gen_cost,
                     (self.ema_disc_cost_real + self.ema_disc_cost_fake) / 2,
                     self.duality_gap,
                    ]
        return np.array(state, dtype='f')

    def check_terminate(self):
        # TODO(haowen)
        # Early stop and recording the best result
        # Episode terminates on two condition:
        # 1) Convergence: inception score doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action

        if self.same_action_count > 30:
            logger.info('Terminate reason: Collapse')
            self.collapse = True
            return True

        step = self.step_number
        if step % self.config.valid_frequency_task == 0:
            self.endurance += 1
            duality_gap = self.get_duality_gap()
            dg = duality_gap
            gen_cost = self.extra_info['gen_cost']
            disc_cost_real = self.extra_info['disc_cost_real']
            disc_cost_fake = self.extra_info['disc_cost_fake']
            decay = self.config.metric_decay
            if self.dg_ema < np.inf:
                self.dg_ema = self.dg_ema * decay + dg * (1 - decay)
            else:
                self.dg_ema = dg

            tf.summary.scalar(' Duality Gap  ',dg,self.step_number)
            tf.summary.scalar(' DG EMA ',self.dg_ema,self.step_number)
            tf.summary.scalar(' DG Reward ',self.compute_reward(self.dg_ema),self.step_number)

            print('\n')
            logger.info('Step: {}, dg: {}'.format(step, dg))
            logger.info('dg_ema: {}'.format(self.dg_ema))
            logger.info('gen_cost: {}'.format(gen_cost))
            logger.info('disc_cost_real: {}, disc_cost_fake: {}'.format(disc_cost_real, disc_cost_fake))
            
            if self.dg_ema < self.best_performance and self.step_number%1000==0:
                self.best_step = self.step_number
                self.best_performance = self.dg_ema
                self.endurance = 0
                self.save_model(step, mute=True)
            
            self.dg_values.append(self.dg_ema)     
            self.generate_images(step)

        if step > self.config.max_training_step:
            logger.info('Terminate reason: Max Training Steps Exceeded')
            return True
        if self.config.stop_strategy_task == 'exceeding_endurance' and \
                self.endurance > self.config.max_endurance_task:
            
            logger.info('Terminate reason: Endurance Exceeded')
            return True
        return False

    def get_step_reward(self, step):
        step = int(step / self.config.valid_frequency_task) - 1
        dg = self.duality_gap
        reward = self.compute_reward(dg)

        if self.metrics_track_baseline[step] == -1:
            self.metrics_track_baseline[step] = dg
            #logger.info(self.metrics_track_baseline)
            return 0

        baseline_dg = self.metrics_track_baseline[step]

        baseline_reward = self.compute_reward(baseline_dg)
        adv = reward - baseline_reward
        # adv = min(adv, self.config.reward_max_value)
        # adv = max(adv, -self.config.reward_max_value)

        decay = self.config.metric_decay
        self.metrics_track_baseline[step] = \
            decay * self.metrics_track_baseline[step] + (1 - decay) * dg
        #logger.info(self.metrics_track_baseline)
        return adv

    def get_final_reward(self):
        if self.collapse:
            return 0, -self.config.reward_max_value
        dg = self.dg_ema
        reward = self.compute_reward(dg)

        # # Calculate baseline
        # if self.reward_baseline is None:
        #     self.reward_baseline = reward
        # else:
        #     decay = self.config.reward_baseline_decay
        #     self.reward_baseline = decay * self.reward_baseline\
        #         + (1 - decay) * reward

        # print('Final Reward : ',reward)
        # # Calcuate advantage
        # adv = reward - self.reward_baseline
        # # # reward_min_value < abs(adv) < reward_max_value
        # adv = math.copysign(max(self.config.reward_min_value, abs(adv)), adv)
        # adv = math.copysign(min(self.config.reward_max_value, abs(adv)), adv)
        # print('Final Reward : ',reward,' ADV  : ',adv)
        
        # if(self.config.adv==True):
        #     return reward, adv
    
        return reward,reward

    def score(self):
    
        config = self.config 
        n_trials   = config.dg_score_ntrials
        batch_size = config.batch_size
        dim_z      = config.dim_z

        scores = []
        for i in range(n_trials):
          X = tf.convert_to_tensor(self.train_dataset.sample_batch(batch_size))
          X = tf.reshape(X,[X.shape[0],-1])
          noise = tf.convert_to_tensor(np.random.normal(0, 1, (batch_size, dim_z)),dtype=tf.float32)
          disc_real = self.dg_discriminator(X,training=False)
          disc_fake = self.dg_discriminator(self.dg_generator(noise,training=False),training=False)
          score = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
        #   score = tf.reduce_mean( tf.math.log(disc_real+1e-4)) + tf.reduce_mean(tf.math.log(abs(1 - disc_fake+1e-4)))
          scores.append(score.numpy())
        return np.average(scores)

    def get_duality_gap(self, splits=None):
        config = self.config

        batch_size = config.batch_size
        dim_z      = config.dim_z
        local_random = config.local_random
        
        if(local_random):
            random_weight_init = [ w + np.random.normal(size=w.shape,scale=config.dg_noise_std).reshape(w.shape) for w in self.discriminator.get_weights()]
            self.dg_discriminator.set_weights(random_weight_init)
        else:    
            self.dg_discriminator.set_weights(self.discriminator.get_weights())
        self.dg_generator.set_weights(self.generator.get_weights())
        
        for step in range(config.dg_train_steps):
            # ----Update DG D network.----
            x = self.train_dataset.sample_batch(batch_size)
            z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)

            self.noise     = z
            self.real_data = tf.reshape(x,[x.shape[0],-1])
            self._dg_train_D()
        
        M1 = self.score()

        if(local_random):
            random_weight_init = [ w + np.random.normal(size=w.shape,scale=config.dg_noise_std).reshape(w.shape) for w in self.generator.get_weights()]
            self.dg_generator.set_weights(random_weight_init)
        else:
            self.dg_generator.set_weights(self.generator.get_weights())
        self.dg_discriminator.set_weights(self.discriminator.get_weights())
        
        for step in range(config.dg_train_steps): 
            # ----Update DG network.----
            z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
            self.noise = z
            self._dg_train_G()

        M2 = self.score()
        
        return abs(M1-M2)

    def compute_reward(self,dg):
        if(self.config.reward_mode=='positive'):
            return self.config.reward_c_positive/(dg+1e-4)

        elif(self.config.reward_mode=='negative'):
            return self.config.reward_c_negative - abs(dg)

        else:
            print(' Unknown Reward Mode : {}'.format(self.config.reward_mode))
            raise NotImplementedError
    
    def evaluate(self, step,action_list,trial=1):

        samples    = self.generator(self.fixed_noise_128).numpy()
        checkpoint_dir = os.path.join('Results',self.config.data_task, self.exp_name,"Trial-{}".format(trial))
        X = self.train_dataset.sample_batch(self.config.vis_count)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # print(samples,X)
        bbox=[-2.5, 2.5, -2.5, 2.5]

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        img_save_path = os.path.join(checkpoint_dir,'output', 'images_{}.png'.format(step))
        kl_save_path = os.path.join(checkpoint_dir,'kl')
        
        
        if not os.path.exists(os.path.join(checkpoint_dir,'output')):
            os.makedirs(os.path.join(checkpoint_dir,'output'))
        
        if not os.path.exists(kl_save_path):
            os.makedirs(kl_save_path)
        



        save_images.save_images(samples.reshape((-1, 28, 28, 1)), img_save_path)

        inception_score = get_inception_score(samples.reshape((-1, 28, 28, 1)))[0].numpy()
        
        fid = get_fid(X.reshape((-1, 28, 28, 1)),samples.reshape((-1, 28, 28, 1)))[0].numpy()
        
        self.inception_score.append(inception_score)
        self.fid.append(fid)

        kl_data = {"INCEPTION_SCORE":self.inception_score,"FID":self.fid}
        

        print(' \n INCEPTION_SCORE ',inception_score)
        print(' \n FID ',fid)
        pd.DataFrame.from_dict(kl_data).to_csv('{}/SCORES.csv'.format(kl_save_path),header=False,index=False)


        if(len(action_list)>0):
            action_list = np.array(action_list)
            action_data = {"G":action_list[:,0],"D":action_list[:,1]}
            pd.DataFrame.from_dict(action_data).to_csv('{}/action_distribution.csv'.format(kl_save_path),header=False,index=False)

    def generate_images(self, step):
        pass


class controller_designed():
    def __init__(self, config=None):
        self.step = 0
        self.config = config

    def sample(self, state):
        self.step += 1
        action = [0, 0]
        action[self.step % 2] = 1
        #disc = self.config.disc_iters
        #gen = self.config.gen_iters
        #if self.step % (disc + gen) < gen:
        #    action[0] =1
        #else:
        #    action[1] = 1
        return np.array(action)

    def initialize_weights(self):
        pass

    def load_model(self, a):
        pass

    def get_weights(self):
        return [0]

    def get_gradients(self, a):
        return [0]

    def train_one_step(self, a, b):
        pass

    def save_model(self, a):
        pass