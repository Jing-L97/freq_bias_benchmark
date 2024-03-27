#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: jliu
'''
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from lm_benchmark.datasets.load_data import get_equal_quantity,get_equal_range,load_data

sns.set_style('whitegrid')

def plot_line(freq_lst: list, score_lst: list, temp: str, model_type: str):
    """plot scatter plot """
    plt.scatter(freq_lst, score_lst, label=str(temp))
    sns.lineplot(freq_lst, score_lst, linewidth=3.5)
    # fit the log curve with error bars
    plt.xlabel('log freq per million in train set', fontsize=15)
    plt.title('Model trained on {} hour audiobook'.format(model_type), fontsize=15, fontweight='bold')
    plt.show()


def plot_bin(root_path:str,model_type:str,y_header:str,num_bins:int,fig_dir:str,mode:str):

    """
    compare effects of different temperatures
    y_header: y-axis header
    multip[le: whether to compare different temperatures
    """
    freq_path = root_path + model_type + '/'
    temp_lst = []
    max_freq = []
    for file in os.listdir(freq_path):
        if not file.startswith('train'):
            temp = file.split('_')[-2]
            binned_freq_frame,freq_frame, max_freq = load_data(freq_path,file,y_header,max_freq,mode,num_bins)
            # print out the annotated groups
            annotated_path = root_path + model_type + '/' + str(num_bins) + '/'
            if not os.path.exists(annotated_path):
                os.makedirs(annotated_path)
            binned_freq_frame.to_csv(annotated_path + mode + '_' + file)
            plot_line(freq_frame['train_Log_norm_freq_per_million'],
                         freq_frame[y_header], temp, model_type)

    plt.xlim(-1, max(max_freq) + 1)
    if y_header == 'score':
        plt.ylim(-1, 1)
        # Plot the boarder line
        plt.plot([-1, max(max_freq)+1], [0, 0], linewidth=3.5, color='red', linestyle='--')
        y_label = y_header
    else:
        plt.ylim(-1, max(max_freq)+1)
        # Plot the diagonal line
        plt.plot([-1, max(max_freq)+1], [-1, max(max_freq)+1], linewidth=3.5, color='red', linestyle='--')
        y_label = 'log freq per million in generation'

    plt.ylabel(y_label, fontsize=15)
    # sort the temp list
    label_lst = ['0.3', '0.6', '1.0', '1.5']
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create a dictionary to map labels to handles
    label_handle_map = dict(zip(labels, handles))
    handles_ordered = [label_handle_map[label] for label in label_lst]
    plt.legend(handles_ordered, label_lst)

    fig_dir = fig_dir + '/' + y_header + '/' + mode + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + model_type + '_' + str(num_bins) + '.png', dpi=800)


root_path = '/data/freq_bias_benchmark/data/generation/gen_freq/inv/'
model_type = '400'
num_bins = 15
mode = 'range'
fig_dir = '/data/freq_bias_benchmark/data/fig/'
y_header = 'score' #'Log_norm_freq_per_million'
plot_bin(root_path,model_type,y_header,num_bins,fig_dir,mode)

def plot_scatter(root_path:str,model_type:str,y_header:str,fig_dir:str):

    """
    compare effects of different temperatures
    y_header: y-axis header
    multip[le: whether to compare different temperatures
    """
    freq_path = root_path + model_type + '/'
    temp_lst = []
    max_freq = []
    for file in os.listdir(freq_path):
        if not file.startswith('train'):
            temp = file.split('_')[-2]
            freq_frame, _, max_freq = get_equal_quantity(freq_path,file,y_header,temp_lst,max_freq,'dot')
            plot_line(freq_frame['train_Log_norm_freq_per_million'],
                         freq_frame[y_header], temp, model_type)
            plt.xlim(-1, max(max_freq))
            plt.ylim(-1, max(max_freq))
            # Plot the diagonal line
            plt.plot([-1, max(max_freq)], [-1, max(max_freq)], linewidth=3.5, color='red', linestyle='--')
            y_label = 'log freq per million in generation'
            plt.ylabel(y_label, fontsize=15)
            # sort the temp list
            plot_dir = fig_dir + '/' + y_header + '/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(plot_dir + model_type + '_' + str(temp) + '.png', dpi=800)
            plt.clf()


root_path = '/data/freq_bias_benchmark/data/generation/gen_freq/oov/'
model_type = '400'
mode = 'dot'
fig_dir = '/data/freq_bias_benchmark/data/fig/'
y_header = 'Log_norm_freq_per_million'
plot_scatter(root_path,model_type,y_header,fig_dir)



def plot_zipf(input_path:str,y_header:str,num_bins:int,fig_dir:str,mode:str):
    """
    Plot word frequency distribution
    input: a list of word freq
    this performs on text iteself without comparison
    """
    # load data
    freq_frame = pd.read_csv(input_path)
    aggregate = False
    if aggregate:
        freq_frame = get_equal_range(freq_frame, y_header, num_bins)
        freq_frame = freq_frame.groupby('group').agg({'Log_freq': 'mean'})
    word_freq = freq_frame['Log_freq'].tolist()
    # Sort word frequencies in descending order
    sorted_word_freq = sorted(word_freq, reverse=True)
    rank_lst = [math.log10(x+1) for x in range(len(sorted_word_freq))]

    # plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rank_lst, sorted_word_freq)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf\'s Law: Word Frequency Distribution')
    plot_dir = fig_dir + '/zipf/' +  '/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + model_type + '_' + str(temp) + '.png', dpi=800)
    plt.show()



# Plot Heap's Law: Vocabulary Growth
vocab_size = []
word_count = 0
unique_words = set()
for word in tokens:
    word_count += 1
    unique_words.add(word)
    vocab_size.append(len(unique_words))


def plot_heaps(input_path):
    """Plot vocab size distribution"""
    
    # load data
    freq_frame = pd.read_csv(input_path)
    aggregate = False
    if aggregate:
        freq_frame = get_equal_range(freq_frame, y_header, num_bins)
        freq_frame = freq_frame.groupby('group').agg({'Log_freq': 'mean'})
    word_freq = freq_frame['Log_freq'].tolist()
    # Sort word frequencies in descending order
    sorted_word_freq = sorted(word_freq, reverse=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(vocab_size)), vocab_size)
    plt.xlabel('Text Length')
    plt.ylabel('Vocabulary Size')
    plt.title('Heap\'s Law: Vocabulary Growth')
    plt.show()


import numpy as np
total_words = len(text.split())
def mutual_information(word1, word2):
    p_word1 = word_freq[word1] / total_words
    p_word2 = word_freq[word2] / total_words
    p_word1_word2 = sum(1 for i in range(len(tokens) - 1) if tokens[i] == word1 and tokens[i + 1] == word2) / total_words
    if p_word1_word2 == 0:
        return 0
    return np.log2(p_word1_word2 / (p_word1 * p_word2))

# Compute mutual information for all word pairs
word_pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
mutual_info_scores = [mutual_information(word1, word2) for word1, word2 in word_pairs]

# Plot the mutual information scores
plt.figure(figsize=(10, 5))
plt.plot(range(len(mutual_info_scores)), mutual_info_scores)
plt.xlabel('Word Pairs')
plt.ylabel('Mutual Information')
plt.title('Mutual Information of Words')
plt.show()

def plot_long_range():



    return
