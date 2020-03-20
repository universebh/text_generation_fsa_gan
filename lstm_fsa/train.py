import json
import argparse

import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from tqdm import tqdm

import config as cfg
from utils import padding
from word2vec_dict import Word2VecDict
from language_model import LanguageModel, init_hidden
from language_dataset import LanguageDataset


############################################################
# Usage of Gensim Word2Vec Object:
#
# target = '<START>'
# target_vec = model_load.wv.__getitem__(['war', 'victory'])
# print(target_vec.shape)
# print(model_load.wv.similar_by_vector(target_vec)[0][0])
# print(model_load.wv.vocab['war'].index)
# print(model_load.wv.index2word[543])
# print(model_load.wv.vector_size)
# print(len(model_load.wv.vocab))
############################################################


# Define config parser
def program_config(parser):
    # ------ Add new params here ------>

    parser.add_argument('--max_seq_len', default=cfg.max_seq_len, type=int)
    parser.add_argument('--test_ratio', default=cfg.test_ratio, type=float)
    parser.add_argument('--hidden_dim', default=cfg.hidden_dim, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--num_epochs', default=cfg.num_epochs, type=int)
    parser.add_argument('--check_interval', default=cfg.check_interval, type=int)
    parser.add_argument('--lr', default=cfg.lr, type=float)
    parser.add_argument('--sch_factor', default=cfg.sch_factor, type=float)
    parser.add_argument('--sch_patience', default=cfg.sch_patience, type=int)
    parser.add_argument('--sch_verbose', default=cfg.sch_verbose, type=bool)
    parser.add_argument('--device', default=cfg.device, type=str)

    parser.add_argument('--emb_model_dir', default=cfg.emb_model_dir, type=str)
    parser.add_argument('--lyrics_dir', default=cfg.lyrics_dir, type=str)
    parser.add_argument('--pretrained_lm_dir', default=cfg.pretrained_lm_dir, type=str)
    parser.add_argument('--save_lm_dir', default=cfg.save_lm_dir, type=str)
    parser.add_argument('--save_tr_l_dir', default=cfg.save_tr_l_dir, type=str)
    parser.add_argument('--save_tr_a_dir', default=cfg.save_tr_a_dir, type=str)
    parser.add_argument('--save_tst_l_dir', default=cfg.save_tst_l_dir, type=str)
    parser.add_argument('--save_tst_a_dir', default=cfg.save_tst_a_dir, type=str)
    parser.add_argument('--save_log_dir', default=cfg.save_log_dir, type=str)

    return parser


# Define training method
def train_dis_epoch(epoch, model, train_loader, criterion, optimizer):
    train_losses, train_accs = [], []
    total_loss, total_acc = 0, 0
    model.train()
    for i, (feature, target) in enumerate(train_loader):
        feature, target = feature.to(cfg.device), target.long().to(cfg.device)
        
        hidden = init_hidden(feature.size(0), cfg.hidden_dim, cfg.device)
        pred = model(feature, hidden)
        pred = pred.view(-1, pred.size(2), pred.size(1))
        
        # pred: batch_size * vocab_size * seq_len
        # target: batch_size * seq_len
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += 100 * torch.sum((pred.argmax(dim=1) == target)).item() / (target.size(0) * target.size(1))
        
        if (i + 1) % cfg.check_interval == 0:
            train_losses.append(total_loss / (i + 1))
            train_accs.append(total_acc / (i + 1))
            cfg.logger.debug(
                "[Epoch %d/%d] [Batch %d/%d] [Train Loss: %f] [Train Acc: %f]"
                % (epoch, cfg.num_epochs, i + 1, len(train_loader), train_losses[-1], train_accs[-1])
            )
            
    cfg.logger.debug(
        "[Epoch %d/%d] [Batch %d/%d] [Train Loss: %f] [Train Acc: %f]"
        % (epoch, cfg.num_epochs, i + 1, len(train_loader), train_losses[-1], train_accs[-1])
    )
    
    return train_losses, train_accs


# Define testing method
def test(model, test_loader, criterion):
    total_loss, total_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for feature, target in tqdm(test_loader, desc='Test'):
            feature, target = feature.to(cfg.device), target.long().to(cfg.device)
            
            hidden = init_hidden(feature.size(0), cfg.hidden_dim, cfg.device)
            pred = model(feature, hidden)
            pred = pred.view(-1, pred.size(2), pred.size(1))
            
            # pred: batch_size * vocab_size * seq_len
            # target: batch_size * seq_len
            loss = criterion(pred, target)
            
            total_loss += loss.item()
            total_acc += 100 * \
            torch.sum((pred.argmax(dim=1) == target)).item() / (target.size(0) * target.size(1))
    
    return total_loss / len(test_loader), total_acc / len(test_loader)


# Main
if __name__ == '__main__':
    # Hyper parameters and configs
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opt = parser.parse_args()
    cfg.init_param(opt)

    # Get word2vec dict with embedding model
    cfg.logger.info('Loading embedding model.')
    wv_dict = Word2VecDict(Word2Vec.load(cfg.emb_model_dir))

    # Load lyrics data, then delete any lyric whose length's greater than max_seq_len
    cfg.logger.info('Loading lyrics data.')
    with open(cfg.lyrics_dir, 'r') as f:
        lyrics_dict = f.read()
        lyrics_dict = json.loads(lyrics_dict)
    data = []
    for key, val in tqdm(lyrics_dict.items()):  # val is a batch    
        cur_seq_len = len(val)
        if cur_seq_len <= cfg.max_seq_len:
            data.append(val)

    # Uncomment this part to train the partial dataset
    # data = data[:100]

    # Split data into training and testing sets
    num_train = int(len(data) * (1 - cfg.test_ratio))
    data_train = data[:num_train]
    data_test = data[num_train:]

    # Torch dataset and dataloader
    train_dataset = LanguageDataset(data_train, wv_dict, padding, cfg.max_seq_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=False)

    if cfg.test_ratio > 0:
        test_dataset = LanguageDataset(data_test, wv_dict, padding, cfg.max_seq_len)
        test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False)

    vocab_size = len(wv_dict.emb_model.wv.vocab) + 1

    # Uncomment this part to check the validity of the dataloader
    # for minibatch in train_loader:
    #     features, targets = minibatch
    #     print(features.size(), targets.size())
    #     for i, (f, t) in enumerate(zip(features, targets)):  # minibatch (one lyric)
    #         for (wv_f, idx_t) in zip(f, t):  # word vector of feature, index of target
    #             print(wv_dict.index2word(wv_dict.vector2index(wv_f.numpy())), wv_dict.index2word(int(idx_t.item())))

    # Print basic info
    cfg.logger.debug('Number of lyrics (Valid / Total): {} / {}'.format(len(data), len(lyrics_dict)))
    cfg.logger.debug('Training / testing size: {} / {}'.format(len(data_train), len(data_test)))
    cfg.logger.debug('Testing set ratio: {}'.format(cfg.test_ratio))
    cfg.logger.debug('Total vocabulary size including paddings: {}'.format(vocab_size))
    cfg.logger.debug('Max sequence length: {}'.format(cfg.max_seq_len))
    cfg.logger.debug('Hidden dimension: {}'.format(cfg.hidden_dim))
    cfg.logger.debug('Batch size: {}'.format(cfg.batch_size))
    cfg.logger.debug('Total epochs: {}'.format(cfg.num_epochs))
    cfg.logger.debug('Intervals to check: {}'.format(cfg.check_interval))
    cfg.logger.debug('Learning rate: {}'.format(cfg.lr))
    cfg.logger.debug('Schedular factor: {}'.format(cfg.sch_factor))
    cfg.logger.debug('Schedular patience: {}'.format(cfg.sch_patience))
    cfg.logger.debug('Schedular verbose: {}'.format(cfg.sch_verbose))
    cfg.logger.debug('Device: {}'.format(cfg.device))
    cfg.logger.debug('Embedding model directory: {}'.format(cfg.emb_model_dir))
    cfg.logger.debug('Lyrics data directory: {}'.format(cfg.lyrics_dir))

    if cfg.pretrained_lm_dir:
        cfg.logger.debug('Pre-trained language model: {}'.format(cfg.pretrained_lm_dir))
    else:
        cfg.logger.debug('Pre-trained language model: initial training')

    # Training
    language_model = LanguageModel(wv_dict, cfg.hidden_dim).to(cfg.device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(language_model.parameters(), lr=cfg.lr)
    schedular = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.sch_factor, patience=cfg.sch_patience, verbose=cfg.sch_verbose)
    if cfg.pretrained_lm_dir:
        lm_loading_res = language_model.load_state_dict(torch.load(cfg.pretrained_lm_dir))
        cfg.logger.debug('Loading language model: {}'.format(lm_loading_res))

    train_losses, train_accs = [], []  # losses & accuracies to save
    if cfg.test_ratio > 0:
        test_losses, test_accs = [], []

    cfg.logger.info('Training.')
    for epoch in range(1, cfg.num_epochs + 1):
        train_losses_, train_accs_ = train_dis_epoch(epoch, language_model, train_loader, criterion, optimizer)
        train_losses += train_losses_
        train_accs += train_accs_
        
        if cfg.test_ratio > 0:
            test_loss_, test_acc_ = test(language_model, test_loader, criterion)
            test_losses.append(test_loss_)
            test_accs.append(test_acc_)
            
            cfg.logger.debug(       
                "[Epoch %d/%d] ----------------> [Test Loss: %f] [Test Acc: %f]"
                % (epoch, cfg.num_epochs, test_losses[-1], test_accs[-1])
            )
        else:
            cfg.logger.debug("-" * 74)

        schedular.step(train_losses[-1])

    # Save language model, losses and training accuracies
    cfg.logger.info('Saving language model.')
    torch.save(language_model.state_dict(), cfg.save_lm_dir)

    cfg.logger.info('Saving training losses.')
    saving_train_losses = pd.DataFrame({'Training Loss': train_losses})
    saving_train_losses.to_csv(cfg.save_tr_l_dir, index=False)
    
    cfg.logger.info('Saving training accuracies.')
    saving_train_accs = pd.DataFrame({'Training Accuracy': train_accs})
    saving_train_accs.to_csv(cfg.save_tr_a_dir, index=False)
    
    if cfg.test_ratio > 0:
        cfg.logger.info('Saving testing losses.')
        saving_test_losses = pd.DataFrame({'Testing Loss': test_losses})
        saving_test_losses.to_csv(cfg.save_tst_l_dir, index=False)

        cfg.logger.info('Saving testing accuracies.')
        saving_test_accs = pd.DataFrame({'Testing Accuracy': test_accs})
        saving_test_accs.to_csv(cfg.save_tst_a_dir, index=False)

    cfg.logger.debug('Saved language model to: {}'.format(cfg.save_lm_dir))

    cfg.logger.debug('Saved training losses to: {}'.format(cfg.save_tr_l_dir))
    cfg.logger.debug('Saved training accuracies to: {}'.format(cfg.save_tr_a_dir))
    if cfg.test_ratio > 0:
        cfg.logger.debug('Saved testing losses to: {}'.format(cfg.save_tst_l_dir))
        cfg.logger.debug('Saved testing accuracies to: {}'.format(cfg.save_tst_a_dir))
    
    cfg.logger.debug('Saved dis training log to: {}'.format(cfg.save_log_dir))
    cfg.logger.info('Everything\'s done.')