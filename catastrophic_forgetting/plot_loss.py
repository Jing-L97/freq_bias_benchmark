#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
plot loss curves during training
@author: jliu
'''

import matplotlib.pyplot as plt

def segment_loss(line:str,loss_dict:dict):
    """segment loss from input dict"""
    records = line.strip().split('|')
    for elem in records:
        if elem.startswith(' loss '):
            loss = float(elem.split(' ')[2])
        if elem.startswith('epoch'):
            epoch = int(elem.split(' ')[1])
    loss_dict[epoch] = loss
    return loss_dict


def load_loss(log_file):
    train_losses = {}
    valid_losses = {}
    with open(log_file, 'r') as f:
        lines = f.readlines()
        cleaned = set(lines)
        for line in cleaned:
            if 'valid on ' in line:
                valid_losses = segment_loss(line,valid_losses)
            elif 'gnorm' in line:     # the smoothed stat of training procedures
                try:
                    train_losses = segment_loss(line,train_losses)
                except:
                    pass

    return dict(sorted(train_losses.items())), dict(sorted(valid_losses.items()))

# Path to the result.log file
log_file = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/model/100_tra.log'
fig_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/fig/loss/'
# Read loss values from the log file
train_losses, valid_losses = load_loss(log_file)

hour = log_file.split('/')[-1].split('.')[0]
# Plot train and validation loss curves
plt.plot(list(train_losses.keys()),list(train_losses.values()), label='Train Loss')
plt.plot(list(valid_losses.keys()), list(valid_losses.values()), label='Validation Loss')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title(f'Train and Validation Loss Curves in {hour}h model', fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(fig_path + hour + '.png', dpi=800)
plt.clf()






