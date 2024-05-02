#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
util func to plot all the figures
@author: jliu
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

color_dict = {'accum':'Orange','ind':'Pink','ood':'Blue','gen_0.3':'Green'
    ,'gen_0.6':'Grey','gen_1.0':'Peru','gen_1.5':'Purple'}

def plot_line(df,x_header:str,y_header:str,label:str,title:str):
    """plot freq score of gen-train comparison"""
    # Plot the line connecting the means
    grouped = df.groupby('group').agg({x_header: 'mean', y_header: 'mean'})
    # Reset index to make 'Group' a column again
    grouped.reset_index(inplace=True)
    # Plot the line connecting the means
    color = color_dict[label]
    plt.plot(grouped[x_header], grouped[y_header], marker='o', linestyle='-', label=label, color=color)

    '''
    for index, row in grouped.iterrows():
        group_data = df[df['group'] == row['group']]
        plt.errorbar(x=row[x_header], y=row[y_header], yerr=group_data[y_header].std())
    '''
    plt.title(title + ' as a function of Token Count', fontsize=15, fontweight='bold')  # Title of the plot
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
    plt.title('Scatter Plot for generated/reference token counts in ' + label + ' corpus', fontsize=15, fontweight='bold')
    plt.xlabel('Reference token counts ', fontsize=12, fontweight='bold')
    plt.ylabel('Generated token counts', fontsize=12, fontweight='bold')
    plt.show()


def get_oov(df,oov_mode):
    """load oov prop of the reference and test corpora"""
    if oov_mode == 'type':
        prop = df[df['Type'] == 'oov'].shape[0] / df.shape[0]
    elif oov_mode == 'token':
        prop = df[df['Type'] == 'oov']['Count_test'].sum() / df['Count_test'].sum()
    return prop








