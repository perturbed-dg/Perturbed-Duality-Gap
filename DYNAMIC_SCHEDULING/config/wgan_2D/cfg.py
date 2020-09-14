import os, sys
import socket
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, root_path)

def get_cfg():
    return Config()

class Config():
    def __init__(self):
        self.hostname = socket.gethostname()
        # Environment & Path
        self.exp_dir = root_path
        
        # Set the path to 2D dataset below
        self.data_task = 'ring'

        exp = 'Vanilla/{}'.format(self.data_task)
        self.data_dir  = './{}/'.format(exp) +'output/gan/2D/'
        self.save_images_dir =  './{}/'.format(exp) +'output/gan/2D/GeneratedPlots/'
        self.model_dir = 'ckpts/2D'
        self.pretrained_mnist_checkpoint_dir =  './{}/'.format(exp) +'weights/mnist_classification' 

        # Task model
        self.dim_z = 100
        self.dim_x = 2
        self.dim_c = 64
        self.disc_iters = 1
        self.n_layers  = 3
        self.h_dim     = 128
        self.gen_iters = 1
        self.dg_batches = 50
        self.dg_splits = 5
        self.activation  = 'relu'
        self.vis_count   = 8000
        self.adv = False
        # Duality Gap
        self.dg_train_steps      = 300
        self.dg_score_ntrials    = 100
        self.local_random        = True
        self.dg_noise_std        = 0.25

        # Training task model
        self.batch_size = 128
        self.lr_task = 5e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.valid_frequency_task = 500
        self.print_frequency_task = 500
        self.stop_strategy_task = 'exceeding_endurance'
        self.max_endurance_task = 5000
        self.max_training_step = 30000

        # Controller
        self.controller_model_name = '2layer_logits_clipping'
        #self.controller_model_name = 'linear_logits_clipping'
        # "How many recent training steps will be recorded"
        self.num_pre_loss = 2

        self.dim_input_ctrl = 4
        self.dim_hidden_ctrl = 128
        self.dim_output_ctrl = 2

        self.reward_baseline_decay = 0.9
        self.reward_c_positive = 5
        self.reward_c_negative = 10

        # Set an max step reward, in case the improvement baseline is too small
        # and cause huge reward.
        self.step_reward      = False
        self.reward_mode      = 'positive' # negative
        self.reward_max_value = 20
        self.reward_min_value = 1
        self.reward_step_ctrl = 0.1
        self.logit_clipping_c = 1

        # Training controller
        self.lr_ctrl = 0.002
        self.total_episodes = 1000
        self.update_frequency_ctrl = 1
        self.print_frequency_ctrl = 100
        self.save_frequency_ctrl = 100
        self.max_endurance_ctrl = 100
        self.rl_method = 'reinforce'
        self.state_decay = 0.9
        self.metric_decay = 0.8

        self.epsilon_greedy = False

    def print_config(self, logger):
        for key, value in vars(self).items():
            logger.info('{}:: {}'.format(key, value))

    def save_config(self, log_path):
        os.makedirs(log_path,exist_ok=True)
        with open(os.path.join(log_path,'cfg.txt'),'a+') as file:
            for key, value in vars(self).items():
                file.write(' {} :: {} \n'.format(key, value))
