def get_target_dim(env_name):
    TARGET_DIM_DICT = {
        "Ant-v5": 27,
        "HalfCheetah-v5": 17,
        "Walker2d-v5": 17,
        "Hopper-v5": 11,
        "Reacher-v5": 11,
        "Humanoid-v5": 292,
        "Swimmer-v5": 8,
        "InvertedDoublePendulum-v5": 11
    }

    return TARGET_DIM_DICT[env_name]

def get_default_steps(env_name):
    if env_name.startswith('HalfCheetah'):
        default_steps = 3000000
    elif env_name.startswith('Hopper'):
        default_steps = 1000000
    elif env_name.startswith('Walker2d'):
        default_steps = 5000000
    elif env_name.startswith('Ant'):
        default_steps = 5000000
    elif env_name.startswith('Swimmer'):
        default_steps = 3000000
    elif env_name.startswith('Humanoid'):
        default_steps = 3000000
    elif env_name.startswith('InvertedDoublePendulum'):
        default_steps = 1000000

    return default_steps


def make_ofe_name(ofe_layer, ofe_unit, ofe_act, ofe_block):
    exp_name = "L{}_U{}_{}_{}".format(ofe_layer, ofe_unit, ofe_act, ofe_block)
    return exp_name
