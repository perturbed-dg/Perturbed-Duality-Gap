""" Decide which loss to update by Reinforce Learning """
# __Author__ == "Haowen Xu"
# __Data__ == "04-07-2018"

import os
import time
import math
import numpy as np
import tensorflow as tf
from models.basic_model import Basic_model
import utils
import os

logger = utils.get_logger()

class Controller(Basic_model):
    def __init__(self, config, exp_name='new_exp_ctrl'):
        super(Controller, self).__init__(config, exp_name)
        
        self._build_graph()


    def _build_graph(self):
        config = self.config
        x_size = config.dim_input_ctrl
        h_size = config.dim_hidden_ctrl
        a_size = config.dim_output_ctrl
        lr     = config.lr_ctrl
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        model_name = config.controller_model_name
        model = tf.keras.models.Sequential()
        
        if model_name == '2layer' or model_name == '2layer_logits_clipping':
            model.add(tf.keras.layers.Dense(h_size,input_dim=x_size))
            model.add(tf.keras.layers.Activation(tf.nn.tanh))
            model.add(tf.keras.layers.Dense(h_size/2))
            model.add(tf.keras.layers.Activation(tf.nn.tanh))
            model.add(tf.keras.layers.Dense(a_size))
            model.add(tf.keras.layers.Activation(tf.nn.softmax))
            
        elif model_name == 'linear' or model_name == 'linear_logits_clipping':
            model.add(tf.keras.layers.Dense(a_size,input_dim=x_size))
            model.add(tf.keras.layers.Activation(tf.nn.softmax))
            
        else:
            raise Exception('Invalid controller_model_name')
        
        model.summary()
        self.model = model
        self.model_init = tf.keras.models.clone_model(self.model)

        self.model.set_weights(self.model_init.get_weights())

        self.saver = tf.train.Checkpoint
        self.models_to_save.append(('controller',self.model,self.optimizer))
        self.models_initial.append((self.model,self.model_init))
        
    def sample(self, state, explore_rate=0.3):
        ######################################################################
        # Sample an action from a given state, probabilistically
        # Args:
        #     state: shape = [dim_input_ctrl]
        #     explore_rate: explore rate

        # Returns:
        #     action: shape = [dim_output_ctrl]
        ######################################################################
    
        a_dist = self.model(np.expand_dims(state,axis=0))
        a_dist = a_dist[0].numpy()
        
        # epsilon-greedy
        if(self.config.epsilon_greedy):
            if np.random.rand() < explore_rate:
                a = np.random.randint(len(a_dist))
            else:
                a = np.argmax(a_dist)

        # continuous
        else:
            a = np.random.choice(a_dist, p=a_dist)
            a = np.argmax(a_dist == a)

        action = np.zeros(len(a_dist), dtype='i')
        action[a] = 1
        return action

    def train_one_step(self, transitions, lr):

    #   transition:
    #       state: shape = [time_steps, dim_input_ctrl]
    #       action: shape = [time_steps, dim_output_ctrl]
    #       reward: shape = [time_steps]
    
        # Retrieve the gradients only for debugging, nothing special.
        
        reward = np.array([trans['reward'] for trans in transitions])
        action = np.array([trans['action'] for trans in transitions])
        state = np.array([trans['state'] for trans in transitions])
        
        with tf.GradientTape() as tape:
            self.reward_plh = reward
            self.action_plh = action
            self.state_plh = state
            
            self.output = self.model(self.state_plh)
            self.chosen_action = tf.argmax(self.output, 1)
            self.action = tf.cast(tf.argmax(self.action_plh, 1), tf.int32)
            self.indexes = tf.range(0, tf.shape(self.output)[0])\
                * tf.shape(self.output)[1] + self.action
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                                self.indexes)
            self.loss = -tf.reduce_mean(tf.math.log(self.responsible_outputs+1e-5)
                                        * self.reward_plh)
            logger.info('Controller Loss : {}'.format(self.loss))
        gradients = tape.gradient(self.loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        return self.loss

    def print_weights(self):
        for idx, var in enumerate(self.model.trainable_variables):
            logger.info('idx:{}, var:{}'.format(idx, var))

    def get_weights(self):
        return self.model.trainable_variables
