import os
import logging

import torch


# Loggers, not a parameter to parse
if not os.path.exists('./log'):
    os.mkdir('./log')
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

# ------ Add new params here ------>

# Hyper parameters
max_seq_len = 32  # only the lyric whose length's less/equal than this value will be used
test_ratio = 0
hidden_dim = 256
batch_size = 128
num_epochs = 50
check_interval = 20
lr = 0.0025
sch_factor = 0.5  # schedular factor
sch_patience = 20  # schedular patients, for ReduceLROnPlateau schedular
sch_verbose = True  # schedular verbose
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Directories
emb_model_dir = './models/word_embedding_models/word2vec.model'
lyrics_dir = './data/ind2line_words.json'

pretrained_lm_dir = './models/language_models/language_model_epoch250.pth'
save_lm_dir = './models/language_models/language_model_epoch300.pth'

save_tr_l_dir = './data/training_losses_epoch300.csv'
save_tr_a_dir = './data/training_accuracies_epoch300.csv'
save_tst_l_dir = './data/nothing_loss.csv'
save_tst_a_dir = './data/nothing_accs.csv'

save_log_dir = './log/train_epoch300.log'


# Init settings according to parser
def init_param(opt):
    # ------ Add new params here ------>

    # logger is not a parameter to parse but need to be global
    global logger, \
        max_seq_len, test_ratio, hidden_dim, batch_size, num_epochs, check_interval, lr, \
        sch_factor, sch_patience, sch_verbose, device, \
        emb_model_dir, lyrics_dir, pretrained_lm_dir, save_lm_dir, \
        save_tr_l_dir, save_tr_a_dir, save_tst_l_dir, save_tst_a_dir, \
        save_log_dir
    
    assert 0.0 <= test_ratio <= 1.0

    # ------ Add new params here ------>

    max_seq_len = opt.max_seq_len
    test_ratio = opt.test_ratio
    hidden_dim = opt.hidden_dim
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    check_interval = opt.check_interval
    lr = opt.lr
    sch_factor = opt.sch_factor
    sch_patience = opt.sch_patience
    sch_verbose = opt.sch_verbose
    device = opt.device

    emb_model_dir = opt.emb_model_dir
    lyrics_dir = opt.lyrics_dir
    pretrained_lm_dir = opt.pretrained_lm_dir
    save_lm_dir = opt.save_lm_dir
    save_tr_l_dir = opt.save_tr_l_dir
    save_tr_a_dir = opt.save_tr_a_dir
    save_tst_l_dir = opt.save_tst_l_dir
    save_tst_a_dir = opt.save_tst_a_dir
    save_log_dir = opt.save_log_dir

    # Set loggers
    fh = logging.FileHandler(save_log_dir, mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Create directories
    logger.info('Create directories if needed.')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./models/language_models'):
        os.mkdir('./models/language_models')


# ------ Add new params to train.py ------>
