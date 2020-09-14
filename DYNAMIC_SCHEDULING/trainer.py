import os
import sys
import math
import socket
import argparse
from time import gmtime, strftime
from collections import deque

import tensorflow as tf
import numpy as np

import utils
from utils.analyse_utils import loss_analyzer_gan
from utils import replaybuffer
from utils import metrics
from models import controller_reinforce
from models import controller_ppo


root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description='run-time arguments', usage='')
    parser.add_argument('--task_name', required=True, type=str, help='task name')
    parser.add_argument('--task_mode', required=True, type=str, help='task mode')
    parser.add_argument('--exp_name', required=True, type=str, help='name of this experiment')
    parser.add_argument('--load_ctrl', default='True',type=str, help='load pretrained controller model')
    parser.add_argument('--save_ctrl', type=bool, default=True, help='whether to save the controller model')
    parser.add_argument('--test_trials', type=int, default=10, help='how many trials to run in test')
    parser.add_argument('--baseline_trials', type=int, default=10, help='how many trials to run in baseline')
    parser.add_argument('--disc_iters', type=int, help='GANs, discrimintor updating iterations')
    parser.add_argument('--gen_iters', type=int, help='GANs, generator updating iterations')
    parser.add_argument('--lr_ctrl', type=float,default=1e-3)
    parser.add_argument('--lr_task', type=float,default=5e-4)
    parser.add_argument('--max_training_step', type=int,default=30000)
    parser.add_argument('--valid_frequency_task', type=int,default=500)
    parser.add_argument('--gpu', type=str,default="0", help='Specify GPU to use')
    parser.add_argument('--model_dir', type=str, help='where to save weights')
    parser.add_argument('--data_dir', type=str, help='data folder')
    parser.add_argument('--data_task', type=str,default='grid', help='Task Dataset')
    parser.add_argument('--activation', type=str,default='relu', help='Task Model Activation')
    parser.add_argument('--reward_mode', type=str, default='positive',help='reward_mode')
    parser.add_argument('--step_reward', type=bool, default=False,help='Accumulate Step Reward')
    parser.add_argument('--epsilon_greedy', type=bool, default=False,help='Exploration')
    parser.add_argument('--adv', type=bool, default=False,help='Use Advantage')
    parser.add_argument('--save_images_dir', type=str, help='where to save images')

    return parser.parse_args()

def arguments_checkout(config, args):
    # ----Task name checking.----
    if not args.task_name in ['wgan_2D','wgan_mnist']:
        logger.exception('Unexcepted task name')
        exit()

    # ----Mode checking.----
    if not args.task_mode in ['train', 'test', 'baseline', 'generate']:
        logger.exception('Unexcepted mode')
        exit()

    # ----Arguments in test mode.----
    if args.task_mode == 'test':
        if args.load_ctrl == 'True':
            args.load_ctrl = os.path.join(config.model_dir,config.data_task, args.exp_name + '_ctrl')
            print(args.load_ctrl)
    
    config.step_reward = args.step_reward
    config.reward_mode = args.reward_mode

def discount_rewards(transitions, final_reward):
    for i in range(len(transitions)):
        transitions[i]['reward'] += final_reward

def display_training_process(task_names,
                             history_train_loss,
                             history_train_acc,
                             history_valid_loss,
                             history_valid_acc):
    for i, task in enumerate(task_names):
        train_loss = np.mean(history_train_loss[i])
        train_acc = np.mean(history_train_acc[i])
        valid_loss = history_valid_loss[i][-1]
        valid_acc = history_valid_acc[i][-1]
        logger.info('Task {:3}, loss: {:8.4f}|{:8.4f}, acc: {:8.4f}|{:8.4f}'.\
                    format(task, train_loss, valid_loss, train_acc, valid_acc))

class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config, args):
        self.config = config

        exp_name = args.exp_name
        logger.info('exp_name: {}'.format(exp_name))

        if config.rl_method == 'reinforce':
            self.model_ctrl = controller_reinforce.Controller(config, exp_name+'_ctrl')
        elif config.rl_method == 'ppo':
            self.model_ctrl = controller_ppo.MlpPPO(config, exp_name+'_ctrl')
        
        elif args.task_name == 'wgan_2D':
            from models import wgan_2D
            self.model_task = wgan_2D.Gan(config, exp_name+'_gan')
        
        elif args.task_name == 'wgan_mnist':
            from models import wgan_mnist
            self.model_task = wgan_mnist.Gan(config, exp_name+'_gan')
           
        else:
            raise NotImplementedError

    def train(self, load_ctrl=True, save_ctrl=True):
        """ Iteratively training between controller and the task model """
        config = self.config
        lr_ctrl = config.lr_ctrl
        lr_task = config.lr_task
        model_ctrl = self.model_ctrl
        model_task = self.model_task
        
        # The smaller the performance (DG) is, the better.
        best_performance = np.inf
        # Record the number of latest episodes without a better result
        endurance = 0

        # ----Initialize controllor.----
        model_ctrl.initialize_weights()
        if load_ctrl:
            model_ctrl.load_model(load_ctrl)

        # ----Start episodes.----
        episode_bar = tf.keras.utils.Progbar(config.total_episodes)

        ctrl_log_dir = 'Logs/{}/{}/Controller/'.format(self.config.data_task,self.model_task.exp_name)
        os.makedirs(ctrl_log_dir,exist_ok=True)
        ctrl_writer = tf.summary.create_file_writer(ctrl_log_dir)

        for ep in range(config.total_episodes):
            
            task_log_dir = 'Logs/{}/{}/Task/Episode-{}'.format(self.config.data_task,self.model_task.exp_name,ep)
            os.makedirs(task_log_dir,exist_ok=True)
            task_writer = tf.summary.create_file_writer(task_log_dir)

            episode_bar.update(ep)
            logger.info('=================')
            logger.info('episodes: {}'.format(ep))

            # ----Initialize task model.----
            model_task.initialize_weights()
            model_task.reset()

            state = model_task.get_state()
            transitions = []
            log_actions = []
            log_rewards = []

            step = -1
            # ----Running one episode.----
            logger.info('TRAINING TASK MODEL...')
            task_bar = tf.keras.utils.Progbar(config.max_training_step)

            with task_writer.as_default():
                while True:
                    step += 1
                    task_bar.update(step)
                    action = model_ctrl.sample(state)
                    state_new, reward, dead = model_task.response(action)
                    extra_info = model_task.extra_info
                    # ----Record training details.----
                    transition = {'state': state,
                                'action': action,
                                'reward': reward,
                                'extra_info': extra_info}
                    #print(transition)
                    transitions.append(transition)
                    log_actions.append(action)
                    log_rewards.append(reward)

                    state = state_new
                    if dead:
                        break
                    task_writer.flush()
            # Reinitialize the controller if the output of the controller
            # collapse to one specific action.

            log_actions = np.array(log_actions)
            log_rewards = np.array(log_rewards)

            if model_task.collapse:
                model_ctrl.initialize_weights()

            # ----Use the best model to get inception score on a
            #     larger number of samples to reduce the variance of reward----
            # if args.task_name == 'gan' and model_task.checkpoint_dir:
            #     # model_task.load_model(model_task.checkpoint_dir)
            #     # TODO use hyperparameter
            #     inps_test = model_task.get_inception_score(5000)
            #     logger.info('inps_test: {}'.format(inps_test))
            #     model_task.update_inception_score(inps_test[0])

            # ----Update the controller.----
            # We actually use the advantage as the final reward
            _, adv = model_task.get_final_reward()
            discount_rewards(transitions, adv)

            if ep % config.update_frequency_ctrl == (config.update_frequency_ctrl - 1):
                logger.info('UPDATING CONTROLLOR...')
                ctrl_loss = model_ctrl.train_one_step(transitions, lr_ctrl)
                logger.info('adv: {}'.format(adv))
                # model_ctrl.print_weights()


            with ctrl_writer.as_default():
                log_discounted_reward = []
                for i in range(len(transitions)):
                    log_discounted_reward.append(transitions[i]['reward'])

                tf.summary.scalar('Final Reward',adv,step=ep)
                tf.summary.scalar('Controller Loss',ctrl_loss,step=ep)
                tf.summary.histogram('Action Distribution',log_actions,step=ep)
                tf.summary.histogram('Step Reward Distribution',log_rewards,step=ep)
                tf.summary.histogram('Discounted Reward Distribution',np.array(log_discounted_reward),step=ep)


            # Printing training results and saving model_ctrl
            save_model_flag = True
            if model_task.best_performance < best_performance:
                best_performance = model_task.best_performance
                save_model_flag = True
                endurance = 0
            else:
                endurance += 1
            logger.info('----LOG FOR EPISODE {}----'.format(ep))
            logger.info('Performance in this episode: {}'.\
                        format(model_task.best_performance))
            logger.info('Best performance till now  : {}'.\
                        format(best_performance))

            if 'gan' in args.task_name:
                loss_analyzer_gan(transitions)
            
            logger.info('--------------------------')

            if save_model_flag and save_ctrl:
                model_ctrl.save_model(ep)
            print(endurance)
            if endurance > config.max_endurance_ctrl:
                break

            ctrl_writer.flush()

    def train_ppo(self, load_ctrl=None, save_ctrl=False):
        config = self.config
        batch_size = config.batch_size
        model_ctrl = self.model_ctrl
        model_task = self.model_task
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBufferMeta = replaybuffer.ReplayBuffer(
            config.buffer_size, keys=keys)
        meta_training_history = deque(maxlen=10)
        epsilons = np.linspace(config.epsilon_start_ctrl,
                               config.epsilon_end_ctrl,
                               config.epsilon_decay_steps_ctrl)

        # ----Initialize controller.----
        model_ctrl.initialize_weights()
        if load_ctrl:
            model_ctrl.load_ctrl(load_ctrl)

        # ----Start meta loop.----
        for ep_meta in range(config.total_episodes):
            logger.info('######## meta_episodes {} #########'.format(ep_meta))

            replayBufferMeta.clear()
            history_train_loss = []
            history_train_acc = []
            history_valid_loss = []
            history_valid_acc = []
            best_acc = []
            best_loss = []
            training_count = []
            history_len_task = config.history_len_task
            for i in range(len(config.task_names)):
                history_train_loss.append(deque(maxlen=history_len_task))
                history_train_acc.append(deque(maxlen=history_len_task))
                history_valid_loss.append(deque(maxlen=history_len_task))
                history_valid_acc.append(deque(maxlen=history_len_task))
                best_acc.append(0)
                best_loss.append(100)
                training_count.append(0)

            # ----Initialize task model.----
            model_task.initialize_weights()
            for i in range(len(config.task_names)):
                valid_loss, valid_acc = model_task.valid(i)
                history_valid_loss[i].append(valid_loss)
                history_valid_acc[i].append(valid_acc)
                history_train_loss[i].append(valid_loss)
                history_train_acc[i].append(valid_acc)

            update_count = 0
            endurance = 0
            transitions = []
            epsilon = epsilons[min(ep_meta, config.epsilon_decay_steps_ctrl - 1)]

            meta_state = model_task.get_state(history_train_loss,
                                              history_train_acc,
                                              history_valid_loss,
                                              history_valid_acc
                                             )

            for step in range(config.max_training_step):
                meta_action = model_ctrl.sample([meta_state], epsilon)[0]
                # From one-hot to index
                meta_action = np.argmax(np.array(meta_action))
                training_count[meta_action] += 1

                state, _, _ = model_task.response(meta_action)
                step_loss = state[1]
                step_acc = state[2]
                history_train_loss[meta_action].append(step_loss)
                history_train_acc[meta_action].append(step_acc)

                meta_state_new = model_task.get_state(history_train_loss,
                                                      history_train_acc,
                                                      history_valid_loss,
                                                      history_valid_acc
                                                      )

                transition = {'state': meta_state,
                              'action': meta_action,
                              'reward': None,
                              'next_state': meta_state_new,
                              'target_value': None}
                transitions.append(transition)

                if step > 0 and step % config.valid_frequency_task == 0:
                    impr_flag = False
                    for i in range(len(config.task_names)):
                        valid_loss, valid_acc = model_task.valid(i)
                        history_valid_loss[i].append(valid_loss)
                        history_valid_acc[i].append(valid_acc)
                        if best_loss[i] > valid_loss:
                            best_loss[i] = valid_loss
                            endurance = 0
                            impr_flag = True
                    if not impr_flag:
                        endurance += 1

                # ----Online update of controller.----
                if step > 0 and step % config.short_horizon_len:
                    update_count += 1
                    meta_reward = model_task.get_step_reward(
                        history_valid_loss, step)
                    for t in transitions:
                        t['reward'] = meta_reward
                        replayBufferMeta.add(t)

                    for _ in range(10):
                        lr = config.lr_ctrl
                        if replayBufferMeta.population == 0:
                            break
                        batch = replayBufferMeta.get_batch(min(config.batch_size_ctrl,
                                                               replayBufferMeta.population))
                        next_value = model_ctrl.get_value(batch['next_state'])
                        gamma = config.gamma_ctrl
                        batch['target_value'] = batch['reward'] + gamma * next_value
                        model_ctrl.train_one_step(batch, lr)

                    if update_count % config.sync_frequency_ctrl == 0:
                        model_ctrl.sync_net()
                        update_count = 0

                # ----Print result.----
                if step % config.display_frequency_task == 0:
                    logger.info('----Step: {:0>6d}----'.format(step))
                    display_training_process(config.task_names,
                                             history_train_loss,
                                             history_train_acc,
                                             history_valid_loss,
                                             history_valid_acc)
                    logger.info('action distribution: {}'.format(
                        np.array(training_count) / sum(training_count)))
                    training_count = [0 for c in training_count]

                meta_state = meta_state_new

                #if self.check_terminate():
                #    break
                if endurance > config.max_endurance_task:
                    logger.info(best_loss)
                    break
            if save_ctrl:
                model_ctrl.save_model(ep_meta)

    def test(self, load_ctrl, ckpt_num=31):
        
        num_trials = 5

        for trial in range(num_trials):

            print("--"*10 +"\n \t TRIAL {} \n".format(trial) +"--"*10)
            config = self.config
            model_ctrl = self.model_ctrl
            model_task = self.model_task
            model_ctrl.initialize_weights()
            model_task.reset()
            # model_task.initialize_weights()
            model_task.init_model()
            model_ctrl.load_model(load_ctrl, ckpt_num=ckpt_num)
            model_task.reset()

            state = model_task.get_state()
            actions = np.array([0, 0])

            self.action_list = []
            
            for i in range(config.max_training_step):
                action = model_ctrl.sample(state)
                actions += np.array(action)
                state_new, _, dead = model_task.response(action)
                state = state_new

                self.action_list.append(action)

                if(i%config.valid_frequency_task == 0):
                    self.model_task.evaluate(i,self.action_list,trial)

            print(actions)

       
    def baseline(self):
        config = self.config
        task_name = config.args.task_name
        if task_name == 'reg':
            from models import reg
            model_ctrl = reg.controller_designed()
        elif task_name == 'cls':
            from models import cls
            model_ctrl = cls.controller_designed()
        elif task_name == 'gan':
            from models import gan
            model_ctrl = gan.controller_designed(config=config)
        elif task_name == 'gan_cifar10':
            from models import gan_cifar10
            model_ctrl = gan_cifar10.controller_designed(config=config)
        elif task_name == 'nmt':
            from models import nmt
            model_ctrl = nmt.controller_designed(config=config)

        if task_name == 'nmt':
            self.config.total_episodes = 1
            self.model_ctrl = model_ctrl
            self.train_ppo()
        else:
            model_task = self.model_task
            model_task.initialize_weights()
            model_task.reset()
            state = model_task.get_state()
            for i in range(config.max_training_step):
                action = model_ctrl.sample(state)
                state, _, dead = model_task.response(action)

        if config.args.task_name == 'reg':
            model_task.load_model()
            loss, _, _ = model_task.valid(model_task.test_dataset)
            logger.info('test_loss: {}'.format(loss))
            return loss
        elif config.args.task_name == 'cls':
            model_task.load_model()
            loss, acc, _, _ = model_task.valid(model_task.test_dataset)
            logger.info('test_loss: {}, test_acc: {}'.\
                        format(loss, acc))
            return acc
        elif config.args.task_name == 'gan':
            model_task.load_model()
            inps = model_task.get_inception_score(500, splits=5)
            logger.info('inception_score_test: {}'.format(inps))
            return inps
        elif config.args.task_name == 'nmt':
            return 0

    def generate(self, load_stud):
        self.model_stud.initialize_weights()
        self.model_stud.load_model(load_stud)
        self.model_stud.generate_images(0)


if __name__ == '__main__':
    # ----Parsing arguments.----
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # ----Loading config file.----
    logger.info('Machine name: {}'.format(socket.gethostname()))
    config_path = os.path.join(root_path, 'config/' + args.task_name)
    config = utils.load_config(config_path)

    arguments_checkout(config, args)
    utils.override_config(config, args)
    config.print_config(logger)
    config.args = args

    # ----Instantiate a trainer object.----
    trainer = Trainer(config, args)

    # ----Main entrance.----
    if args.task_mode == 'train':
        # ----Training----
        logger.info('TRAIN')
        if config.rl_method == 'reinforce':
            trainer.train(load_ctrl=args.load_ctrl, save_ctrl=args.save_ctrl)
        elif config.rl_method == 'ppo':
            trainer.train_ppo(load_ctrl=args.load_ctrl, save_ctrl=args.save_ctrl)
    elif args.task_mode == 'test':
        ## ----Testing----
        logger.info('TEST')
        test_metrics = []
        for i in range(args.test_trials):
            test_metrics.append(trainer.test(args.load_ctrl))
        logger.info(test_metrics)
    elif args.task_mode == 'baseline':
        # ----Baseline----
        logger.info('BASELINE')
        baseline_metrics = []
        for i in range(args.baseline_trials):
            baseline_metrics.append(trainer.baseline())
        logger.info(baseline_metrics)
    elif args.task_mode == 'generate':
        # ----Generate---- only in gan tasks
        logger.info('GENERATE')
        trainer.generate(args.load_stud)
