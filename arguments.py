import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="PPO") #训练使用的策略
    parser.add_argument("--env", default="HalfCheetah-v5")#训练环境
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--steps", default=100000, type=int)#训练总步数，默认10W
    parser.add_argument("--sac-units", default=256, type=int)#SAC网络的单位数，默认256
    parser.add_argument("--batch_size", default=256, type=int)#每次训练的批量大小，默认为 256
    parser.add_argument("--gin", default= "./gins/meta.gin")
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true",
                        help="remove existed directory")
    parser.add_argument("--dir-root", default="output_PPO", type=str)
    parser.add_argument("--img_save_dir", default='./results/img')

    # fourier
    parser.add_argument("--discount", default=0.99, type=float)#discount: 折扣因子，默认值为 0.99
    parser.add_argument("--dim_discretize", default=128, type=int)#dim_discretize: 离散化维度，默认值为 128
    parser.add_argument("--fourier_type", default='dtft', type=str)#设置傅里叶变换类型，默认为 dtft（离散时间傅里叶变换）
    parser.add_argument("--normalizer", default='layer', type=str, choices=['layer', 'batch'])#设置正则化方法，默认使用 layer，也可以选择 batch


    # loss
    parser.add_argument("--use_projection", default=True, action="store_true")# 是否使用投影操作，默认为 True
    parser.add_argument("--projection_dim", default=512, type=int)#投影维度，默认值为 512
    parser.add_argument("--cosine_similarity", default=True, action="store_true")#是否使用余弦相似度，默认为 True

    # auxiliary records
    parser.add_argument("--qval_img", default=False, action="store_true")
    parser.add_argument("--tsne", default=False, action="store_true")
    parser.add_argument("--record_grad", default=False, action="store_true")#是否记录梯度信息，默认为 False
    parser.add_argument("--save_model", default=True, action="store_true")
    parser.add_argument("--record_state", default=False, action="store_true")
    # parser.add_argument("--visual_buffer", default=False, action="store_true")
    # parser.add_argument("--record_rb_ind", default=False, action="store_true")

    # intervals
    parser.add_argument("--summary_freq", default=1000, type=int) #每多少步记录一次摘要（训练进度），默认为 1000
    parser.add_argument("--eval_freq", default=5000, type=int) #每多少步进行一次评估，默认为 5000
    parser.add_argument("--value_eval_freq", default=20000, type=int) #每多少步进行一次值函数的评估，默认为 20000
    parser.add_argument("--sne_freq", default=10000, type=int)
    # parser.add_argument("--record_state_freq", default=500, type=int)
    parser.add_argument("--grad_freq", default=10000, type=int)  #
    parser.add_argument("--random_collect", default=4000, type=int)
    parser.add_argument("--pre_train_step", default=200, type=int)
    parser.add_argument("--save_freq", default=10000, type=int)
    # parser.add_argument("--visual_buffer_freq", default=5000, type=int)
    
    # target
    parser.add_argument("--target_update_freq", default=100, type=int)#目标网络的更新频率，默认为 100
    parser.add_argument("--tau", default=0.01, type=float)#tau: 软更新的步长，默认为 0.01

    # ppo
    parser.add_argument("--steps_per_epoch", default=2000, type=int)#steps_per_epoch: 每个 epoch 训练的步数，默认为 4000
    parser.add_argument("--lam", default=0.97, type=float)#用于 GAE（广义优势估计）的方法参数，默认为 0.97
    parser.add_argument("--update_every", default=5, type=int)#每多少步更新一次模型，默认为 5

    # evaluate when estimating maml loss
    parser.add_argument("--evaluate_every", default=20, type=int)

    parser.add_argument("--remark", default="HalfCheetah, SPF")

    # get_data
    parser.add_argument("--aux", default="raw", type=str, choices=['raw', 'OFE', 'FSP'])


    # output dim
    parser.add_argument("--dim_output", default=128, type=int, help="Dimension of the output")#设置输出层的维度，默认为 128

    args = parser.parse_args()

    return args