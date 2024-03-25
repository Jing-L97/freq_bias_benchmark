#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: jliu
'''
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('whitegrid')

def get_equal_bins(data_frame, col_header:str, n_bins:int):
    '''
    get equal-sized bins
    input: a sorted array or a list of numbers; computes a split of the data into n_bins bins of approximately the same size
    return
        bins: array with each bin boundary
        data_frame: updated df with an additional column of group
    '''
    # sort the dataframe
    data_frame = data_frame.sort_values(by=col_header)
    data = data_frame[col_header]
    # preparing data (adding small jitter to remove ties)
    size = len(data)
    assert n_bins <= size, "too many bins compared to data size"
    mindif = np.min(np.abs(np.diff(np.sort(np.unique(data)))))  # minimum difference between consecutive distinct values
    jitter = mindif * 0.01  # this small jitter will not change the relative order between datapoints
    data_jitter = np.array(data) + np.random.uniform(low=-jitter, high=jitter, size=size)
    data_sorted = np.sort(data_jitter)  # little jitter to remove ties

    # Creating the bins with approx equal number of observations
    bin_indices = np.linspace(1, len(data), n_bins + 1) - 1  # indices to edges in sorted data
    bins = [data_sorted[0]]  # left edge inclusive
    bins = np.append(bins, [(data_sorted[int(b)] + data_sorted[int(b + 1)]) / 2 for b in bin_indices[1:-1]])
    bins = np.append(bins, data_sorted[-1] + jitter)  # this is because the extreme right edge is inclusive in plt.hits

    # computing bin membership for the original data; append bin membership to stat
    bin_membership = np.zeros(size, dtype=int)
    for i in range(0, len(bins) - 1):
        bin_membership[(data_jitter >= bins[i]) & (data_jitter < bins[i + 1])] = i
    data_frame['group'] = bin_membership
    return data_frame


# re-calculate bins by same range of each bin
def load_data(freq_path,file,y_header,temp_lst,max_freq,mode):
    """load data to plot figures"""
    freq_frame = pd.read_csv(freq_path + file)
    temp = file.split('_')[-2]
    temp_lst.append(float(temp))
     # remove oov words
    freq_frame = freq_frame[freq_frame['score']!='oov']
    freq_frame['train_Log_norm_freq_per_million'] = freq_frame['train_Log_norm_freq_per_million'].astype(float)
    freq_frame[y_header] = freq_frame[y_header].astype(float)
    if mode == 'bin':
        freq_frame = get_equal_bins(freq_frame,'train_Log_norm_freq_per_million', num_bins)
        freq_frame = freq_frame.groupby('group').agg({'train_Log_norm_freq_per_million': 'mean',
                                                          y_header: 'mean'})
    max_freq.append(freq_frame['train_Log_norm_freq_per_million'].max())
    return freq_frame,temp_lst,max_freq


def plot_line(freq_lst: list, score_lst: list, temp: str, model_type: str):
    """plot scatter plot """
    plt.scatter(freq_lst, score_lst, label=str(temp))
    sns.lineplot(freq_lst, score_lst, linewidth=3.5)
    # fit the log curve with error bars
    plt.xlabel('log freq per million in train set', fontsize=15)
    plt.title('Model trained on {} hour audiobook'.format(model_type), fontsize=15, fontweight='bold')
    plt.show()



def plot_inv(root_path:str,model_type:str,y_header:str,num_bins:int,fig_dir:str,mode:str):

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
            freq_frame, temp_lst, max_freq = load_data(freq_path,file,y_header,temp_lst,max_freq,mode)
            plot_line(freq_frame['train_Log_norm_freq_per_million'],
                         freq_frame[y_header], temp, model_type)

    plt.xlim(-1, max(max_freq))
    if y_header == 'score':
        plt.ylim(-1, 1)
        # Plot the boarder line
        plt.plot([-1, max(max_freq)], [0, 0], linewidth=3.5, color='red', linestyle='--')
        y_label = y_header
    else:
        plt.ylim(-1, max(max_freq))
        # Plot the diagonal line
        plt.plot([-1, max(max_freq)], [-1, max(max_freq)], linewidth=3.5, color='red', linestyle='--')
        y_label = 'log freq per million in generation'

    plt.ylabel(y_label, fontsize=15)
    # sort the temp list
    label_lst = ['0.3', '0.6', '1.0', '1.5']
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create a dictionary to map labels to handles
    label_handle_map = dict(zip(labels, handles))
    handles_ordered = [label_handle_map[label] for label in label_lst]
    plt.legend(handles_ordered, label_lst)

    fig_dir = fig_dir + '/' + y_header + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + model_type + '_' + str(num_bins) + '.png', dpi=800)


root_path = '/data/freq_bias_benchmark/data/generation/gen_freq/inv/'
model_type = '400'
num_bins = 2
mode = 'bin'
fig_dir = '/data/freq_bias_benchmark/data/fig/'
y_header =  'score' #'''Log_norm_freq_per_million'
plot_inv(root_path,model_type,y_header,num_bins,fig_dir,mode)

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
            freq_frame, _, max_freq = load_data(freq_path,file,y_header,temp_lst,max_freq,'dot')
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
