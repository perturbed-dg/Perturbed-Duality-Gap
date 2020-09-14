
## Requirements
```
python>=3.6, tensorflow-gpu==2.2.0, matplotlib==2.2.2, imageio==2.4.1, sklearn==0.20.2
```

## Getting Started

Install and Activate a Virtual Environment
```
virtualenv --python=python3.6 ./env
source ./env/bin/activate
```
Install Dependencies
```
pip install -r requirements.txt
```

## Datasets


### GANs
Our GANs experiments use Synthetic 2D Mixture of Gaussian, [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) datasets.
Set a desired path for the argument `data_dir` in config file `config/gan_2D/cfg.py`. 
The training scripts will automatically download the data at its first run.


The CIFAR-10 data is available at [Download page of CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html).
You need to download the python version from this page and unzip it. Set the argument 'data\_dir' in config file 'config/gan_cifar.py' to where you save the data.

## Experiments
Use the following script to launch an experiment:

```
python trainer.py
    --task_name=[task_name] 
    --task_mode=[task_mode] 
    --exp_name=[exp_name]
```
where
- `task_name` is one of:  `wgan_2D`,`wgan_mnist`.
- `task_mode` is one of: `train`, `test`.
- `exp_name` can be any string you would like use to name this experiment.

### Example: GAN 2D 
Train a controller on the WGAN-2D task:

```
python trainer.py --task_name=wgan_2D --task_mode=train --exp_name=DG
```

After the training is done, we use the trained controller to guide the training of the task model on a new dataset:
```
# Make sure the `exp_name` is set as the one used in the controller training experiment. 
python trainer.py --task_name=wgan_2D --task_mode=test --exp_name=DG
```
The script will automatically load the controller trained in the `DG` experiment.

Alternatively, you can specify a specific checkpoint you want to test with by:
```
python trainer.py --task_name=gan_2D --task_mode=test --exp_name=DG --load_ctrl=/path/to/checkpoint/folder/
```
