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
    parser.add_argument('--ngram', type=int, default=5,
                        help='ngram to extract')
    parser.add_argument('--mode', type=str, default='quantity',
                        help='which type of words to select; recep or exp')
    parser.add_argument('--plot_type', type=str, default='oov',
                        help='which type of words to select; recep or exp')
    parser.add_argument('--oov_mode', type=str, default='type',
                        help='which type of oov to plot; type or token')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='num_bins to ')
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



'''

# plot ttr
model_type = '400'
input_root = ('/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/gen_freq/inv/400/')
fig_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/fig/'
n_gram_lst = [1,2,3,4,5]
#n_gram_lst = [2]
for n_gram in n_gram_lst:
    #plot_distinct_n(input_root, fig_dir, n_gram, model_type)
    compare_zipf(input_root, fig_dir, n_gram, model_type)


'''




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
    elif plot_type == 'missing':
        plot_missing(df, x_header, mode, num_bins, file)
    # save the fig
    if not plot_type == 'oov':
        plt.savefig(fig_path + plot_type + '.png', dpi=800)

    plt.legend()



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





