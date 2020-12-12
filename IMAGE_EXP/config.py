
LATENT_DEPTH = 100

BATCH_SIZE   = 32
NUM_EPOCHS   = 50

GPU          = 0
MODEL_NAME   = "DCGAN"
N_CRITIC     = 1
N_GENERATOR  = 1
CLIP_CONST   = 0.05
LR_D         = 6e-4
LR_G         = 1e-4

SETTING      = "divergence"
DATASET      = "mnist"
PERTUB_STD   = 0.1


DYNAMIC_NOISE     = False
CLASSIC_REFERENCE = True
LOG_FREQ          = 500

HYPARAMS = {
    "mnist": {
        "project_shape": [7, 7, 256],
        "gen_filters_list": [128, 64, 1],
        "gen_strides_list": [1, 2, 2],
        "disc_filters_list": [64, 128],
        "disc_strides_list": [2, 2]
    },
    "fashion_mnist": {
        "project_shape": [7, 7, 256],
        "gen_filters_list": [128, 64, 1],
        "gen_strides_list": [1, 2, 2],
        "disc_filters_list": [64, 128],
        "disc_strides_list": [2, 2]
    },
    "cifar10": {
        "project_shape": [4, 4, 256],
        "gen_filters_list": [64,64,64,3],
        "gen_strides_list": [2,2,1,2],
        "disc_filters_list": [64,64,64],
        "disc_strides_list": [2,2,2]
    },
    "celeb_a": {
       "project_shape": [4, 4, 256],
        "gen_filters_list": [64,64,64,3],
        "gen_strides_list": [2,2,1,2],
        "disc_filters_list": [64,64,64],
        "disc_strides_list": [2,2,2]
    }
}
