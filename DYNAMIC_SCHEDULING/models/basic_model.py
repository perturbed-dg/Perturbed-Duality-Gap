""" Basic model object """
# __Author__ == "Haowen Xu"
# __Data__ == "05-04-2018"

import os
import math
import time

import numpy as np
import tensorflow as tf

import utils
from models import layers

logger = utils.get_logger()

class Basic_model():
    def __init__(self, config, exp_name='new_exp'):
        self.config = config
        self.exp_name = exp_name
        self.models_to_save = []
        self.models_initial= []
        self.checkpoint_dir = os.path.join(self.config.model_dir,self.config.data_task, exp_name)
        self.config.save_config(self.checkpoint_dir)
        
    def reset(self):
        raise NotImplementedError

    def _build_placeholder(self):
        raise NotImplementedError

    def _build_graph(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def load_model(self, checkpoint_dir=None, ckpt_num=None):
        
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        
        os.makedirs(checkpoint_dir,exist_ok=True)
        for model_name,model,optimizer in self.models_to_save:
            model_checkpoint_dir = os.path.join(self.checkpoint_dir,model_name)
            model_checkpoint_prefix = os.path.join(model_checkpoint_dir,model_name+'step_{}_ckpt'.format(ckpt_num))


            if not os.path.exists(model_checkpoint_dir):
                return
            
            checkpoint = self.saver(optimizer=optimizer, model=model)        
            status = checkpoint.restore(tf.train.latest_checkpoint(model_checkpoint_dir))
            logger.info('===Restoring {} : {} === \n '.format(model_checkpoint_prefix,status))
            
    def save_model(self, step, mute=False):
        
        for model_name,model,optimizer in self.models_to_save:
            model_checkpoint_dir = os.path.join(self.checkpoint_dir,model_name)
            model_checkpoint_prefix = os.path.join(model_checkpoint_dir,model_name+'step_{}_ckpt'.format(step))

            if not os.path.exists(model_checkpoint_dir):
                os.makedirs(model_checkpoint_dir,exist_ok=True)
            
            if not mute:
                logger.info('Save model at {}'.format(model_checkpoint_dir))

            checkpoint = self.saver(optimizer=optimizer, model=model)
            checkpoint.save(file_prefix=model_checkpoint_prefix)

    def print_weights(self, tvars=None):
        if not tvars:
            tvars = self.tvars
        for tvar in tvars:
            logger.info('===begin===')
            logger.info(tvar)
            logger.info(self.sess.run(tvar))
            logger.info('===end===')

    def get_grads_magnitude(self, grads):
        v = []
        for grad in grads:
            v.append(np.reshape(grad, [-1]))
        v = np.concatenate(v)
        return np.linalg.norm(v) / np.sqrt(v.shape[0])

    def init_model(self):
        self._build_graph()
        self.reset()
        
    def initialize_weights(self):
        for model,model_init in self.models_initial:
            model.set_weights(model_init.get_weights())