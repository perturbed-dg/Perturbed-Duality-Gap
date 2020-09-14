
LATENT_DEPTH = 100

BATCH_SIZE   = 128
NUM_EPOCHS   = 30


MODEL_SAVE_DIR = "./Results"

GPU          = 1
MODEL_NAME   = "WGAN_GP"
N_CRITIC     = 5
N_GENERATOR  = 1
CLIP_CONST   = 2.0
LR_D         = 5e-1
LR_G         = 5e-5

SETTING      = "divergence"
DATASET      = "celeb_a"
PERTUB_STD   = 0.75


DYNAMIC_NOISE     = False
CLASSIC_REFERENCE = True
LOG_FREQ = 300

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
