import tensorflow as tf
mnist = tf.keras.datasets.mnist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# from models.wgan_mnist import *
from models.wgan_2D import *
import utils
from utils.analyse_utils import loss_analyzer_reg
from utils.analyse_utils import loss_analyzer_gan
from utils import replaybuffer
from utils import metrics

config_path = os.path.join('./', 'config/wgan_2D' )
config = utils.load_config(config_path)
gan = Gan(config)
gan.train()