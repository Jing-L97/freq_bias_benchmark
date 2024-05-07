#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
util func to plot all the figures
@author: jliu
'''
import matplotlib.pyplot as plt
import seaborn as sns
import enchant
d = enchant.Dict("en")

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

def is_word(word):
    # Function to check if a word is valid
    try:
        return d.check(word)
    except:
        return False

def get_oov(df):
    """load oov prop of the reference and test corpora"""
    type_prop = df[df['Type'] == 'oov'].shape[0] / df.shape[0]
    token_prop = df[df['Type'] == 'oov']['Count_test'].sum() / df['Count_test'].sum()
    true_type_prop =
    false_type_prop =
    return type_prop,token_prop,true_prop,false_prop





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







