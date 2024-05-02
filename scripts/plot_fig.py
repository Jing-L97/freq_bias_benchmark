#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
plot different figures recursively
@author: jliu
'''

import argparse
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from lm_benchmark.load_data import load_data
from lm_benchmark.plot_util import plot_line, plot_scatter, get_oov
sns.set_style('whitegrid')

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--root_path', type=str, default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/',
                        help='root path to the utterance and freq dir')
    parser.add_argument('--model', type=str, default='400',
                        help='model name')
    parser.add_argument('--ngram', type=int, default=4,
                        help='ngram to extract')
    parser.add_argument('--mode', type=str, default='quantity',
                        help='which type of words to select; recep or exp')
    parser.add_argument('--plot_type', type=str, default='missing',
                        help='which type of words to select; recep or exp')
    parser.add_argument('--oov_mode', type=str, default='type',
                        help='which type of oov to plot; type or token')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='which type of oov to plot; type or token')
    return parser.parse_args(argv)


def plot_missing(df,x_header:str, mode:str, num_bins:int,file:str):
    """plot prop/prob of missing words in each group"""
    # segment into groups
    df = df[df['Type'] != 'oov']
    freq_frame = load_data(df, x_header, mode, num_bins)
    if file.startswith('accum'):
        frame_all = freq_frame
    if not file.startswith('accum'):
        # append the prop column for each bin
        freq_framed = freq_frame.groupby('group')
        # loop and get prop
        frame_all = pd.DataFrame()
        for _, freq_group in freq_framed:
            prop = freq_group[freq_group['Type'] == 'missing'].shape[0] / freq_group.shape[0]
            # append the results
            freq_group['p_miss'] = prop
            # return the new freq_frame with the appended prop columnn
            frame_all = pd.concat([frame_all, freq_group])

    y_header = 'p_miss'
    plot_line(frame_all, x_header, y_header, file[:-4], 'p_miss')


def plot_score(df,x_header:str, mode:str, num_bins:int,file:str):
    """plot freq score of the reference and test corpora"""
    df = df[df['Type'] != 'oov']     # use the whole reference corpus
    # select non-oov words
    freq_frame = load_data(df, x_header, mode, num_bins)
    y_header = 'Score'
    title = 'Freq Score'
    plot_line(freq_frame, x_header, y_header, file[:-4], title)
    plt.ylim(-1, 1)


def plot_freq(df,x_header:str, mode:str, num_bins:int,file:str,max_freq:list):
    """plot freq of the reference and test corpora"""
    y_header = 'Count_test'
    df = df[df['Type'] == 'inv']
    # select inv words
    freq_frame = load_data(df, x_header, mode, num_bins)
    title = 'Generation Token Count'
    plot_line(freq_frame, x_header, y_header, file[:-4], title)
    plt.yscale("log")
    max_freq.append(df['Count_test'].max())
    max_freq.append(df['Count_ref'].max())

def plot_freq_scatter(df,fig_path:str,file:str):
    """plot freq scatter points of the reference and test corpora"""
    df = df[df['Type'] == 'inv']
    # select inv words
    plot_scatter(df['Count_ref'], df['Count_test'], file[:-4])
    plt.savefig(fig_path + 'scatter_' + file[:-4] + '.png', dpi=800)



def plot_oov(data:dict,oov_mode:str,fig_path:str):
    # Sort the dictionary based on the given order
    given_order = ['ind', 'ood', 'gen_0.3', 'gen_0.6', 'gen_1.0', 'gen_1.5']
    sorted_data = {key: data[key] for key in given_order}
    # Extract x and y values from the sorted dictionary
    x_values = list(sorted_data.keys())
    y_values = list(sorted_data.values())
    # Plotting
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel('Test sets', fontsize=12, fontweight='bold')
    plt.ylabel('Prop of oov ' + oov_mode, fontsize=12, fontweight='bold')
    plt.title('Prop of oov ' + oov_mode + ' in different test sets', fontsize=15, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(fig_path + 'oov_' + oov_mode +'.png', dpi=800)



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


'''

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


'''




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



def main(argv):
    # load args
    args = parseArgs(argv)
    ngram = args.ngram
    oov_mode = args.oov_mode
    model = args.model
    freq_path = args.root_path + 'freq' + '/' + model + '/' + str(ngram) + '_gram/'
    fig_path = args.root_path + 'fig' + '/' + model + '/' + str(ngram) + '_gram/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    x_header = 'Count_ref'
    num_bins = args.num_bins
    mode = args.mode
    plot_type = args.plot_type

    plt.clf()
    max_freq = []
    prop_dict = {}
    for file in os.listdir(freq_path):
        if file.endswith('csv'):
            try:
                df = pd.read_csv(freq_path + file)
                if plot_type == 'score':
                    plot_score(df, x_header, mode, num_bins, file)

                elif plot_type == 'freq':      # the accum model will be passed due to the missing test corpus
                    plot_freq(df, x_header, mode, num_bins, file,max_freq)

                elif plot_type == 'scatter':
                    plot_freq_scatter(df, fig_path, file)

                elif plot_type == 'missing':
                    plot_missing(df, x_header, mode, num_bins, file)

                elif plot_type == 'oov':    # get oov type prop num from each
                    prop_dict[file[:-4]] = get_oov(df,oov_mode)
            except:
                print(file)

    if plot_type == 'score':
        plt.axhline(y=0, color='red', linewidth=3.5, linestyle='--', label='y = 0')
    elif plot_type == 'freq':
        plt.plot([0, max(max_freq)], [0, max(max_freq)], linewidth=3.5, color='red', linestyle='--')
    elif plot_type == 'oov':
        plot_oov(prop_dict,oov_mode,fig_path)
    plt.legend()
    # save the fig
    if not plot_type == 'oov':
        plt.savefig(fig_path + plot_type + '.png', dpi=800)





if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





