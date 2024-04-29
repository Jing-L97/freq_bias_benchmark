#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
util func to plot all the figures
@author: jliu
'''

import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from lm_benchmark.load_data import get_equal_range,load_data
import numpy as np
sns.set_style('whitegrid')


def plot_line(df,x_header:str,y_header:str,label:str,title:str):
    """plot freq score of gen-train comparison"""
    # Plot the line connecting the means
    grouped = df.groupby('group').agg({x_header: 'mean', y_header: 'mean'})
    # Reset index to make 'Group' a column again
    grouped.reset_index(inplace=True)
    # Plot the line connecting the means
    plt.plot(grouped[x_header], grouped[y_header], marker='o',linestyle='-',label = label)
    '''
    for index, row in grouped.iterrows():
        group_data = df[df['group'] == row['group']]
        plt.errorbar(x=row[x_header], y=row[y_header], yerr=group_data[y_header].std())
    '''
    plt.title(title + ' as a function of Token Count')  # Title of the plot
    plt.xlabel('Token Count (in reference corpus)')  # Label for the x-axis
    plt.ylabel(title)  # Label for the y-axis
    plt.grid(True)  # Show grid lines
    plt.xscale("log")




def plot_scatter(ref_counts, gen_counts,label:str):
    # display a scatterplot of log count generated versus log count in the reference corpus
    plt.figure(figsize=(10, 10))
    plt.scatter(ref_counts, gen_counts)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot([0, max(ref_counts)], [0, max(ref_counts)], 'r-')
    # Adding title and labels
    plt.title('Scatter Plot for generated/reference token counts in ' + label + ' corpus')
    plt.xlabel('Reference token counts ')
    plt.ylabel('Generated token counts')
    plt.show()

def missing_prob(df):
    """plot prop of missing words in each group"""
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    plt.plot(df['Count_ref'], df['p_miss'], label='p_miss', color='blue')  # Plot the data
    plt.title('prob of missing as a function of Token Count for the accumulator model')  # Title of the plot
    plt.xlabel('Token Count (in ref corpus)')  # Label for the x-axis
    plt.ylabel('Probability of missing (in gen corpus)')  # Label for the y-axis
    plt.ylim(0, 1)
    plt.xscale("log")
    plt.grid(True)  # Show grid lines
    plt.legend()  # Show legend






def freq_scatter(ref_counts, gen_counts,label:str):
    # display a scatterplot of log count generated versus log count in the reference corpus
    plt.figure(figsize=(10, 10))
    plt.scatter(ref_counts, gen_counts)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot([0, max(ref_counts)], [0, max(ref_counts)], 'r-')
    # Adding title and labels
    plt.title('Scatter Plot for generated/reference token counts in ' + label + ' corpus')
    plt.xlabel('Reference token counts ')
    plt.ylabel('Generated token counts')
    plt.show()







def compare_zipf(input_root, fig_dir, n_gram, model_type):
    def plot_zipf(input_path:str,y_header:str,num_bins:int,mode:str,label:str):
        """
        Plot word frequency distribution
        input: a list of word freq
        this performs on text iteself without comparison
        """
        # load data
        freq_frame = pd.read_csv(input_path)
        if mode == 'range':
            freq_frame = get_equal_range(freq_frame, y_header, num_bins)

        freq_frame = freq_frame.groupby('group').agg({'Log_freq': 'mean'})
        word_freq = freq_frame['Log_freq'].tolist()
        # Sort word frequencies in descending order
        sorted_word_freq = sorted(word_freq, reverse=True)
        rank_lst = [math.log10(x+1) for x in range(len(sorted_word_freq))]
        # plot results
        plt.plot(rank_lst, sorted_word_freq,linewidth = 3.5, label = label)
        plt.ylim(0, 6)
        plt.xlabel('Rank')
        plt.ylabel('Frequency')

    num_bins = 20
    mode = 'range'

    for file in os.listdir(input_root + str(n_gram) + '_gram/'):
        if file.endswith('.csv'):
            label = file.split('_')[-2]
            input_path = input_root + str(n_gram) + '_gram/' + file
            # change into different headers
            y_header = 'Log_norm_freq_per_million'
            plot_zipf(input_path,y_header,num_bins,mode,label)
            plt.title('Zipf\'s Law: ' + str(n_gram) + '_gram')
    # Specify the desired order of legend labels
    legend_order = ['train', '0.3', '0.6', '1.0', '1.5']
    # Get the handles and labels of the current axes
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create a dictionary to map labels to handles
    label_to_handle = dict(zip(labels, handles))
    # Create sorted handles list based on the desired order
    handles_sorted = [label_to_handle[label] for label in legend_order]
    # Create the legend with sorted handles and specified labels
    plt.legend(handles_sorted, legend_order, loc='best')
    plot_dir = fig_dir + '/zipf/' + model_type + '/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_dir + str(n_gram) + '.png', dpi=800)
    plt.legend()
    plt.clf()



def plot_distinct_n(input_root,fig_dir,n_gram,model_type):
    #input_root = input_root + str(model_type) + '/'
    label_lst = ['train','0.3', '0.6', '1.0', '1.5']
    plot_dir = fig_dir + 'distinct_n/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    ttr_lst = []
    for label in label_lst:
            # change into different headers
            if label == 'train':
                frame = pd.read_csv(input_root + str(n_gram) + '_gram/train_400.csv')
            else:
                frame = pd.read_csv(input_root + str(n_gram) + '_gram/unprompted_'+label+'_400.csv')
            ttr = frame.shape[0]/frame['Freq'].sum()
            ttr_lst.append(ttr)
    sns.lineplot(x = label_lst, y = ttr_lst, linewidth=3.5,label = 'n = ' + str(n_gram))
    plt.xlabel('temperature', fontsize=15)
    plt.ylabel('ratio', fontsize=15)
    plt.ylim(0, 1)
    plt.title('Distinct-n across different temperatures', fontsize=15, fontweight='bold')
    plt.savefig(plot_dir + str(model_type) + '.png', dpi=800)



# plot ttr
model_type = '400'
input_root = ('/Users/jli'
              'u/PycharmProjects/freq_bias_benchmark/data/generation/gen_freq/inv/400/')
fig_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/fig/'
n_gram_lst = [1,2,3,4,5]
#n_gram_lst = [2]
for n_gram in n_gram_lst:
    #plot_distinct_n(input_root, fig_dir, n_gram, model_type)
    compare_zipf(input_root, fig_dir, n_gram, model_type)

def plot_distr(data:list,temp:str,num_bins:int,mode:str):
    """plot distr of selected words"""
    # Plot the number distribution
    data = sorted(data)
    # Compute histogram
    if mode == 'range':

        counts, bin_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data) + 1))
        # Plot the curve
        plt.plot(bin_edges[:-1], counts, marker='o', linestyle='-',label = temp)

    elif mode == 'quantity':
        '''
        bins = pd.qcut(data, q=num_bins, duplicates='drop', labels=False)
        bin_counts = np.bincount(bins)
        # Plot the histogram
        plt.plot(range(len(bin_counts)), bin_counts, marker='o', linestyle='-', label=temp)
        '''
        counts, bin_edges = np.histogram(data, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
        plt.plot(bin_centers, counts, marker='o', linestyle='-', label=temp)

    plt.legend()



def plot_missing_freq(root_path:str,temp_lst:list,fig_dir:str,mode:str):
    """plot missing words in the train set"""
    all_freq = pd.read_csv(root_path)
    # select the missing words in differnt temp
    for temp in temp_lst:
        selected_freq = all_freq[all_freq[temp + '_Log_norm_freq_per_million'] == -5000]
        # plot the corresponding distr
        plot_distr(selected_freq['Log_norm_freq_per_million'],temp,num_bins,mode)
    # Add labels and title
    plt.xlabel('Log_norm_freq_per_million in train set', fontsize=12, fontweight='bold')
    plt.ylabel('Missing words Count', fontsize=12, fontweight='bold')
    plt.title('Missing words freq distribution in train set', fontsize=15, fontweight='bold')
    plt.xlim(-1, 4.5)
    # Set figure size
    plt.gcf().set_size_inches(10, 4)
    plt.savefig(fig_dir + mode +'.png', dpi=800)
    plt.clf()

# select the missing words in different temp
root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/gen_freq/inv/400/1_gram/matched.csv'
temp_lst = ['0.3', '0.6', '1.0', '1.5']
num_bins = 20
fig_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/fig/missing_words/'
mode = 'quantity'
plot_missing_freq(root_path,temp_lst,fig_dir,mode)

def plot_oov(root_path:str,freq_path:str,temp_lst:list,fig_dir:str,mode:str,set_type:str):
    """plot the oov words freq in the generation set"""
    ref_freq = pd.read_csv(freq_path + set_type + '.csv')
    # loop and get oov frames
    for temp in temp_lst:
        all_freq = pd.read_csv(root_path+temp+'_400.csv')
        selected_words = all_freq[all_freq['score'] == 'oov']['Word']
        selected_freq = ref_freq[ref_freq['Word'].isin(selected_words)]
        # match the corresponding words in the reference set
        plot_distr(selected_freq['Log_norm_freq_per_million'],temp,num_bins,mode)
    # Set figure size
    # Add labels and title
    plt.xlabel('Log_norm_freq_per_million in ' +  set_type + ' set', fontsize=12, fontweight='bold')
    plt.ylabel('OOV words count', fontsize=12, fontweight='bold')

    plt.ylim(0, 6000)
    plt.xlim(-1, 4.5)


    plt.title('OOV words freq distribution in ' + set_type + ' set', fontsize=15, fontweight='bold')
    plt.gcf().set_size_inches(10, 4)
    plt.savefig(fig_dir + set_type + '_' + mode +'.png', dpi=800)
    plt.clf()



