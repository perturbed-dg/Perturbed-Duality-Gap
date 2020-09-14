import os
import random

import numpy as np
import tensorflow as tf

from models.basic_model import Basic_model
import utils
logger = utils.get_logger()

class BasePPO(Basic_model):
    def __init__(self, config, exp_name):
        super(BasePPO, self).__init__(config, exp_name)
        self.update_steps = 0

    def _build_placeholder(self):
        raise NotImplementedError

    def _build_graph(self):
        
        self.actor     = self.build_actor_net()
        self.actor_old = self.build_actor_net()

        self.critic = self.build_critic_net()

        self.optimizer1 = tf.keras.optimizers.Adam(self.config.lr_ctrl)
        self.optimizer2 = tf.keras.optimizers.Adam(self.config.lr_ctrl)
        
        self.saver = tf.train.Checkpoint

    def build_actor_net(self):
        raise NotImplementedError

    def sample(self, states, epsilon=0.5):
        dim_a = self.config.dim_output_ctrl
        action = np.zeros(dim_a, dtype='i')
        
        if random.random() < epsilon:
            a = random.randint(0, dim_a - 1)
            action[a] = 1
            return action, 'random'
        else:
            pi = self.actor(states)[0]
            #a = np.argmax(pi)
            a = np.random.choice(dim_a, 1, p=pi)[0]
            action[a] = 1
            return action, pi

    def sync_net(self):
        self.actor_old.set_weights(self.actor.get_weights())
        logger.info('{}: target_network synchronized'.format(self.exp_name))

    def update_critic(self, transition_batch, lr=0.001):
        
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        target_value = transition_batch['target_value']
        
        with tf.GradientTape() as tape:
            pi     = self.actor(state) + 1e-8
            old_pi = self.actor_old(state) + 1e-8

            value = self.critic(state)

            a_indices = tf.stack([tf.range(tf.shape(action)[0], dtype=tf.int32),action], axis=1)
            pi_wrt_a = tf.gather_nd(params=pi, indices=a_indices, name='pi_wrt_a')
            old_pi_wrt_a = tf.gather_nd(params=old_pi, indices=a_indices,
                                        name='old_pi_wrt_a')

            cliprange = self.config.cliprange_ctrl
            gamma = self.config.gamma_ctrl

            adv = (target_value - value)
            adv = tf.stop_gradient(adv, name='critic_adv_stop_gradient')
        
            critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.critic.trainable_variables])
            mse_loss = tf.reduce_mean(tf.square(target_value - value))
        
            reg_param = 0.0
            self.critic_loss = mse_loss + reg_param * critic_reg_loss

        gradients = tape.gradient(self.critic_loss,self.critic.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients,self.critic.trainable_variables))
        
    def update_actor(self, transition_batch, lr=0.001):
        self.update_steps += 1
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        target_value = transition_batch['target_value']
        
        with tf.GradientTape() as tape:
            pi     = self.actor(state) + 1e-8
            old_pi = self.actor_old(state) + 1e-8

            value = self.critic(state)

            a_indices = tf.stack([tf.range(tf.shape(action)[0], dtype=tf.int32),action], axis=1)
            pi_wrt_a = tf.gather_nd(params=pi, indices=a_indices, name='pi_wrt_a')
            old_pi_wrt_a = tf.gather_nd(params=old_pi, indices=a_indices,
                                        name='old_pi_wrt_a')

            cliprange = self.config.cliprange_ctrl
            gamma = self.config.gamma_ctrl

            adv = (target_value - value)
            adv = tf.stop_gradient(adv, name='critic_adv_stop_gradient')

            ratio = pi_wrt_a / old_pi_wrt_a
            pg_losses1 = adv * ratio
            pg_losses2 = adv * tf.clip_by_value(ratio,
                                                1.0 - cliprange,
                                                1.0 + cliprange)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(pi * tf.log(pi), 1))
            actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.actor.trainable_variables])
            pg_loss = -tf.reduce_mean(tf.minimum(pg_losses1, pg_losses2))
            beta = self.config.entropy_bonus_beta_ctrl
            reg_param = 0.0

            self.actor_loss = pg_loss + beta * entropy_loss + reg_param * actor_reg_loss
        
        gradients = tape.gradient(self.actor_loss,self.actor.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients,self.actor.trainable_variables))

        self.pi = pi
        self.value = value
        self.ratio = ratio

        
    def train_one_step(self, transition_batch, lr=0.001):
        self.update_actor(transition_batch, lr)
        self.update_critic(transition_batch, lr)

    def get_value(self, state):
        value = self.critic(state)
        return value

    def print_weights(self):
        pass
    def get_weights(self):
        pass


class MlpPPO(BasePPO):
    def __init__(self, config, exp_name='MlpPPO'):
        super(MlpPPO, self).__init__(config, exp_name)
        self._build_placeholder()
        self._build_graph()

    def _build_placeholder(self):
        config = self.config
        
    def build_actor_net(self):
        
        dim_h = self.config.dim_hidden_ctrl
        dim_a = self.config.dim_output_ctrl
        dim_x = self.config.dim_input_ctrl
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(dim_h,input_dim=dim_x))
        model.add(tf.keras.layers.Activation(tf.nn.tanh))
        
        model.add(tf.keras.layers.Dense(dim_a))
        model.add(tf.keras.layers.Activation(tf.nn.softmax))
        
        model.summary()
        return model

    def build_critic_net(self):
        dim_h = self.config.dim_hidden_ctrl
        dim_a = self.config.dim_output_ctrl
        dim_x = self.config.dim_input_ctrl
        
        model = tf.keras.models.Sequential()
       
        model.add(tf.keras.layers.Dense(dim_h,input_dim=dim_x))
        model.add(tf.keras.layers.Activation(tf.nn.tanh))
       
        model.add(tf.keras.layers.Dense(1))
        return model
