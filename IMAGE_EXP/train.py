import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import config as conf
import matplotlib.pyplot as plt
from models.gan import GAN
from models.nsgan import NSGAN
from models.dcgan import DCGAN
from models.wgan_gp import WGAN_GP
from pprint import pprint
from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(conf.GPU)

MODELS = {"WGAN_GP":WGAN_GP,"GAN":GAN,"NSGAN":NSGAN,"DCGAN":DCGAN}


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def feature_normalize(features, feature_depth):
    return (features/255 - 0.5) / 0.5


def feature_denormalize(features, feature_shape):
    return (features + 1) / 2

def get_data_generator(data_path,batch_size,image_size):
        images = []
        for dirname, dirnames, filenames in os.walk(data_path):
            images += [os.path.join(dirname, f) for f in filenames]

        train_images,test_images = train_test_split(images)
        test_images,eval_images  = train_test_split(test_images)

        nbatch = int(np.ceil(len(images) / batch_size))
        return  DataGenerator(train_images,image_size=image_size,batch_size=batch_size),DataGenerator(test_images,image_size=image_size,batch_size=batch_size),DataGenerator(eval_images,image_size=image_size,batch_size=batch_size)

def main():
    
    save_path = './EXP/{}/{}/{}'.format(conf.DATASET,conf.MODEL_NAME,conf.SETTING)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open('{}/config.txt'.format(save_path), 'wt') as out:
        pprint(vars(conf), stream=out)

    print('Saving To : {}'.format(save_path))

    pprint(vars(conf))

    logdir   = os.path.join(save_path,'Logs')
    os.makedirs(logdir,exist_ok=True) 
    writer = tf.summary.create_file_writer(logdir)

    hyparams = conf.HYPARAMS[conf.DATASET]

    latent_depth = conf.LATENT_DEPTH

    batch_size = conf.BATCH_SIZE
    num_epochs = conf.NUM_EPOCHS

    n_critic   = conf.N_CRITIC
    n_generator= conf.N_GENERATOR
    clip_const = conf.CLIP_CONST
    log_freq   = conf.LOG_FREQ
    
    dynamic_noise     = conf.DYNAMIC_NOISE
    classic_reference = conf.CLASSIC_REFERENCE

    print(conf)

    
    if(conf.DATASET=='celeb_a_'):
        data_dir = './../SAGAN-tensorflow2.0/Dataset/CELEBA/'
        train_loader,test_loader,eval_loader = get_data_generator(data_dir,batch_size,32)
        train_loader = train_loader.generator()
        test_loader = test_loader.generator()
        eval_loader = eval_loader.generator()
        num_sets      = 150000
        feature_shape = 3
        feature_depth = 3

    else:
        train_loader, info = tfds.load(conf.DATASET,split="train[:80%]",with_info=True, shuffle_files=True,download=True,data_dir='./data')
        test_loader,_ = tfds.load(conf.DATASET,split="train[80%:90%]",with_info=True, shuffle_files=True,download=True,data_dir='./data')
        eval_loader,_ = tfds.load(conf.DATASET,split="train[90%:]",with_info=True, shuffle_files=True,download=True,data_dir='./data')

        img_dim  = 28
        if(conf.DATASET=='celeb_a_' or conf.DATASET=='cifar10'):
            img_dim  = 32

        train_loader= train_loader.map(lambda x: {"image":(tf.image.resize(x['image'],[img_dim,img_dim]))}).repeat().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_loader = test_loader.map(lambda x: {"image":(tf.image.resize(x['image'],[img_dim,img_dim]))}).repeat().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        eval_loader = eval_loader.map(lambda x: {"image":(tf.image.resize(x['image'],[img_dim,img_dim]))}).repeat().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        num_sets      = info.splits["train"].num_examples
        feature_shape = info.features["image"].shape
        feature_depth = np.prod(feature_shape)

    model = MODELS[conf.MODEL_NAME](
        project_shape=hyparams["project_shape"],
        gen_filters_list=hyparams["gen_filters_list"],
        gen_strides_list=hyparams["gen_strides_list"],
        disc_filters_list=hyparams["disc_filters_list"],
        disc_strides_list=hyparams["disc_strides_list"],
        setting = conf.SETTING
    )

    dg_model = MODELS[conf.MODEL_NAME](
        project_shape=hyparams["project_shape"],
        gen_filters_list=hyparams["gen_filters_list"],
        gen_strides_list=hyparams["gen_strides_list"],
        disc_filters_list=hyparams["disc_filters_list"],
        disc_strides_list=hyparams["disc_strides_list"],
        setting = conf.SETTING
    )
    
    D_norm = []
    G_norm = []
    D_var = []
    G_var = []

    lr_d = conf.LR_D
    lr_g = conf.LR_G

    if('WGAN' in conf.MODEL_NAME):
        generator_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_g)
        discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_d)

        dg_generator_opt = tf.keras.optimizers.RMSprop(lr_g)
        dg_discriminator_opt = tf.keras.optimizers.RMSprop(lr_d)

    else:
        generator_opt     = tf.keras.optimizers.Adam(lr_g)
        discriminator_opt = tf.keras.optimizers.Adam(lr_d)
        
        dg_generator_opt = tf.keras.optimizers.Adam(lr_g)
        dg_discriminator_opt = tf.keras.optimizers.Adam(lr_d)

    if(conf.SETTING=='mode_collapse'):
        generator_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_g)
        discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_d)

    def classic_discriminator_loss(real_output, fake_output):
        if('WGAN' in conf.MODEL_NAME):
            real_output = tf.math.sigmoid(real_output)
            fake_output = tf.math.sigmoid(fake_output)

        real_loss  = tf.reduce_mean(tf.math.log(real_output+1e-4))
        fake_loss  = tf.reduce_mean(tf.math.log(1 - fake_output + 1e-4))
        total_loss = -(real_loss + fake_loss)/2

        return total_loss

    def classic_generator_loss(fake_output):
        if( 'WGAN' in conf.MODEL_NAME):
            fake_output = tf.math.sigmoid(fake_output)
        return tf.reduce_mean(tf.math.log(1 - fake_output + 1e-4))


    @tf.function
    def train_disc_step(x, z):
        with tf.GradientTape() as discriminator_tape:
            discriminator_loss = model.discriminator_loss(x, z)
            
            grads_discriminator_loss = discriminator_tape.gradient(
                target=discriminator_loss, sources=model.discriminator.trainable_variables
            )

            discriminator_opt.apply_gradients(
                zip(grads_discriminator_loss, model.discriminator.trainable_variables)
            )
        
            if('WGAN' in conf.MODEL_NAME):
                for w in model.discriminator.trainable_variables:
                    w.assign(tf.clip_by_value(w, -clip_const, clip_const))
        
        return discriminator_loss
    
    @tf.function
    def dg_train_disc_step(x, z,classic_reference=False):
        with tf.GradientTape() as discriminator_tape:

            if(classic_reference):
                discriminator_loss = classic_discriminator_loss(dg_model.discriminator(x),dg_model.discriminator(dg_model.generator(z)))
            else:
                discriminator_loss = dg_model.discriminator_loss(x, z)
            
            grads_discriminator_loss = discriminator_tape.gradient(
                target=discriminator_loss, sources=dg_model.discriminator.trainable_variables
            )

            dg_discriminator_opt.apply_gradients(
                zip(grads_discriminator_loss, dg_model.discriminator.trainable_variables)
            )

            if('WGAN' in conf.MODEL_NAME):
                for w in dg_model.discriminator.trainable_variables:
                    w.assign(tf.clip_by_value(w, -clip_const, clip_const))

        return discriminator_loss
    
    
    @tf.function
    def train_gen_step(z):
        with tf.GradientTape() as generator_tape:
            generator_loss = model.generator_loss(z)

            grads_generator_loss = generator_tape.gradient(
                target=generator_loss, sources=model.generator.trainable_variables
            )

            generator_opt.apply_gradients(
                zip(grads_generator_loss, model.generator.trainable_variables)
            )

        return generator_loss

    @tf.function
    def dg_train_gen_step(z,classic_reference=False):
        with tf.GradientTape() as generator_tape:
            
            if(classic_reference):
                generator_loss = classic_generator_loss(dg_model.discriminator(dg_model.generator(z)))
            else:
                generator_loss = dg_model.generator_loss(z)

            grads_generator_loss = generator_tape.gradient(
                target=generator_loss, sources=dg_model.generator.trainable_variables
            )

            dg_generator_opt.apply_gradients(
                zip(grads_generator_loss, dg_model.generator.trainable_variables)
            )

        return generator_loss


    steps_per_epoch = num_sets // batch_size
    train_steps = steps_per_epoch * num_epochs

    generator_losses = []
    discriminator_losses = []
    generator_losses_epoch = []
    discriminator_losses_epoch = []
    x_fakes = []
    
    DG = {
        'vanilla':[],
        'local_random':[]
    }
    
    M1 = {
        'vanilla':[],
        'local_random':[]
    }
    
    M2 = {
        'vanilla':[],
        'local_random':[]
    }
    
    def score():
        epochs = 200
        scores = []
        for _ in range(epochs):
            x = next(eval_iterator)
            x_i = feature_normalize(x["image"], feature_depth)
            z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)

            if(classic_reference):
                scores.append(-1*classic_discriminator_loss(dg_model.discriminator(x_i), dg_model.discriminator(dg_model.generator(z_i))))
            else:
                scores.append(-1*dg_model.discriminator_loss(x_i, z_i))
        return np.average(scores)
    
    def compute_duality_gap():
        M_u_v_worst = 0
        M_u_worst_v = 0
        epochs      = 300
        
        g_opt_weight = [tf.zeros_like(var) for var in dg_generator_opt.get_weights()]
        d_opt_weight = [tf.zeros_like(var) for var in dg_discriminator_opt.get_weights()]
        
        progbar = tf.keras.utils.Progbar(epochs)
        
        dg_generator_opt.set_weights(g_opt_weight)
        dg_discriminator_opt.set_weights(d_opt_weight)

        dg_model.generator.set_weights(model.generator.get_weights())    
        dg_model.discriminator.set_weights(model.discriminator.get_weights())
        for e in range(epochs):
            progbar.update(e)
            z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
            generator_loss_i = dg_train_gen_step(z_i)
          
        M_u_worst_v =  score()
        
        dg_model.generator.set_weights(model.generator.get_weights())    
        dg_model.discriminator.set_weights(model.discriminator.get_weights())
        progbar = tf.keras.utils.Progbar(epochs)
        for e in range(epochs):
            progbar.update(e)
            x = next(test_iterator)
            x_i = feature_normalize(x["image"], feature_depth)
            z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
            discriminator_loss_i = dg_train_disc_step(x_i, z_i)

        
        M_u_v_worst =  score()
        
        DG['vanilla'].append(abs(M_u_v_worst - M_u_worst_v))
        M1['vanilla'].append(M_u_v_worst)
        M2['vanilla'].append(M_u_worst_v)
        
        
        if(dynamic_noise):
            random_weight_init = [ w + np.random.normal(size=w.shape,scale=np.sqrt(np.asarray(D_var)[-1,i])).reshape(w.shape) for i,w in enumerate(model.discriminator.get_weights())]
        else:
            random_weight_init = [ w + np.random.normal(size=w.shape,scale=conf.PERTUB_STD).reshape(w.shape) for i,w in enumerate(model.discriminator.get_weights())]


        dg_generator_opt.set_weights(g_opt_weight)
        dg_discriminator_opt.set_weights(d_opt_weight)

        dg_model.discriminator.set_weights(random_weight_init)
        dg_model.generator.set_weights(model.generator.get_weights())
        progbar = tf.keras.utils.Progbar(epochs)
        for e in range(epochs):
            progbar.update(e)
            x = next(test_iterator)
            x_i = feature_normalize(x["image"], feature_depth)
            z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
            discriminator_loss_i = dg_train_disc_step(x_i, z_i)
        M_u_v_worst =  score()
        
        if(dynamic_noise):
            random_weight_init = [ w + np.random.normal(size=w.shape,scale=np.sqrt(np.asarray(G_var)[-1,i])).reshape(w.shape) for i,w in enumerate(model.generator.get_weights())]
        else:
            random_weight_init = [ w + np.random.normal(size=w.shape,scale=conf.PERTUB_STD).reshape(w.shape) for i,w in enumerate(model.generator.get_weights())]
        
        dg_model.generator.set_weights(random_weight_init)
        dg_model.discriminator.set_weights(model.discriminator.get_weights())
        progbar = tf.keras.utils.Progbar(epochs)
        for e in range(epochs):
            progbar.update(e)
            z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
            generator_loss_i = dg_train_gen_step(z_i)
        
        M_u_worst_v =  score()
        
        
        DG['local_random'].append(abs(M_u_v_worst - M_u_worst_v))
        M1['local_random'].append(M_u_v_worst)
        M2['local_random'].append(M_u_worst_v)
        
        print('Dulaity Gaps : \n\t Vanilla :{} \n\t Random : {}\n'.format(DG['vanilla'][-1],DG['local_random'][-1]))
    

    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)
    eval_iterator = iter(eval_loader)
    
    x = next(train_iterator)
    x_i = feature_normalize(x["image"], feature_depth)
    z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
    
    dg_train_disc_step(x_i, z_i)
    dg_train_gen_step(z_i)

    dg_model.discriminator.summary()
    dg_model.generator.summary()
    
    z_vis = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
    
    for i in range(0, train_steps+1):
        
        epoch = i // steps_per_epoch
        iteration = i
        print("Epoch: %i ====> %i / %i" % (epoch+1, i % steps_per_epoch, steps_per_epoch), end="\r")
        
        x = next(train_iterator)
        x_i = feature_normalize(x["image"], feature_depth)
        z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)
        
        for _ in range(n_critic):
            discriminator_loss_i = train_disc_step(x_i, z_i)
        discriminator_losses.append(discriminator_loss_i)
        
        for _ in range(n_generator):
            generator_loss_i = train_gen_step(z_i)
        generator_losses.append(generator_loss_i)

        
        if i % steps_per_epoch == 0:
            x_fake = model.generator(z_i, training=False)
            x_fake = feature_denormalize(x_fake, feature_shape)

            generator_loss_epoch = np.mean(generator_losses[-steps_per_epoch//n_critic:])
            discriminator_loss_epoch = np.mean(discriminator_losses[-steps_per_epoch:])

            print("Epoch: %i,  Generator Loss: %f,  Discriminator Loss: %f" % \
                (epoch, generator_loss_epoch, discriminator_loss_epoch)
            )

            generator_losses_epoch.append(generator_loss_epoch)
            discriminator_losses_epoch.append(discriminator_loss_epoch)

            x_fakes.append(x_fake)
        
      
        if i % log_freq == 0:
            
            d_weights = model.discriminator.get_weights()
            g_weights = model.generator.get_weights()

            d_norm = [ np.linalg.norm(x) for x in d_weights ]
            g_norm = [ np.linalg.norm(x) for x in g_weights ]

            d_var  = [np.var(x) for x in d_weights]
            g_var  = [np.var(x) for x in g_weights]

            D_norm.append(d_norm)
            G_norm.append(g_norm)
            D_var.append(d_var)
            G_var.append(g_var)
            compute_duality_gap()
            
            save_data = {
                    'DG':DG,
                    'M1':M1,
                    'M2':M2,
                    'D_norm':np.asarray(D_norm),
                    'G_norm':np.asarray(G_norm),
                    'D_var':np.asarray(D_var),
                    'G_var':np.asarray(G_var),
                    'D_Losses':discriminator_losses,
                    'G_Losses':generator_losses,
                    'X_Fakes':x_fakes
                }
                
            with open( os.path.join(save_path,'data.pkl'), 'wb') as fp:
                pickle.dump(save_data, fp)
            
            r, c = 5, 5
            noise = z_vis
            gen_imgs = model.generator(noise).numpy()
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(r, c)
            cnt = 0

            if(conf.DATASET=='cifar10' or conf.DATASET=='celeb_a'):
                for i in range(r):
                    for j in range(c):
                        axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                        axs[i,j].axis('off')
                        cnt += 1
            else:        
                for i in range(r):
                    for j in range(c):
                        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                        axs[i,j].axis('off')
                        cnt += 1
            
            os.makedirs('{}/images'.format(save_path),exist_ok=True)
            fig.savefig("{}/images/{}.png".format(save_path,epoch))
            plt.close()
           
            weight_save_path  = '{}/weights'.format(save_path)
            generator_weights = '{}/weights/G/step_{}/'.format(save_path,iteration)
            discriminator_weights = '{}/weights/D/step_{}/'.format(save_path,iteration)
            os.makedirs(generator_weights,exist_ok = True)               
            os.makedirs(discriminator_weights,exist_ok = True)               
            model.generator.save(generator_weights)
            model.discriminator.save(discriminator_weights)
      
            r, c = 3, 2
            fig, axs = plt.subplots(r, c,figsize=(16,16))
            cnt = 0
            dg_x_axis = [i for i in range(len(DG['vanilla']))]
            
            axs[0,1].set_title('M1')
            
            _DG_ = {}
            _DG_['vanilla']      = pd.DataFrame(DG['vanilla']).ewm(alpha=0.1).mean()
            _DG_['local_random'] = pd.DataFrame(DG['local_random']).ewm(alpha=0.1).mean()
            
            axs[0,1].plot(dg_x_axis,M1['vanilla'],color='r',label='Vanilla')
            axs[0,1].plot(dg_x_axis,M1['local_random'], color='b',label='Local Random')

            axs[0,0].set_title('Duality Gap')
            axs[0,0].plot(dg_x_axis,_DG_['vanilla'],color='r',label='Vanilla')
            axs[0,0].plot(dg_x_axis,_DG_['local_random'], color='b',label='Ours')
            axs[0,0].legend()

            norm_axis = [i*log_freq for i in range(len(D_norm))]
            var_axis = [i*log_freq for i in range(len(D_var))]

            axs[1,0].set_title('Discriminator Weight Norm')
            for i in range(len(model.discriminator.get_weights())):
                axs[1,0].plot(norm_axis,np.asarray(D_norm)[:,i],label='Layer {}'.format(i+1))

            axs[1,1].set_title('Discriminator Weight Variance')
            for i in range(len(model.discriminator.get_weights())):
                axs[1,1].plot(var_axis,np.asarray(D_var)[:,i],label='Layer {}'.format(i+1))


            axs[2,0].set_title('Generator Weight Norm')
            for i in range(len(model.generator.get_weights())):
                axs[2,0].plot(norm_axis,np.asarray(G_norm)[:,i],label='Layer {}'.format(i+1))

            axs[2,1].set_title('Generator Weight Variance')
            for i in range(len(model.generator.get_weights())):
                axs[2,1].plot(var_axis,np.asarray(G_var)[:,i],label='Layer {}'.format(i+1))


#             plt.legend()

            fig_dir = '{}/dg_plots'.format(save_path)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir,exist_ok=True)
            plt.savefig(fig_dir+'/{}.png'.format(iteration))
            plt.close()        

if __name__ == "__main__":
    main()