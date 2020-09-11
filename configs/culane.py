# DATA
dataset='CULane'
#data_root = None
data_root='/home/zty/CULane'

# TRAIN
epoch = 50
batch_size = 16
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '34'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/zty/Ultra-Fast-Lane-Detection-master/log'

# FINETUNE or RESUME MODEL PATH
finetune = None
#resume = '/home/zty/Ultra-Fast-Lane-Detection-master/log/20200715_214332_lr_2e-01_b_16/ep0.pth'
resume = None

# TEST
test_model = '/home/zty/Lane-detection/weight/res34.pth'
test_work_dir = None





