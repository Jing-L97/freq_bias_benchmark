"""
Construct same-sized datasets in/out of the domain
"""

import os
import pandas as pd
from tqdm import tqdm
from lm_benchmark.datasets.load_data import txt2csv

filename_path = '/data/freq_bias_benchmark/data/train/filename/7100.csv'
train_freq_dir = '/data/freq_bias_benchmark/data/train/train_freq/'
text_dir = '/data/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/'
out_dir = '/data/freq_bias_benchmark/data/train/oov/train_utt/'


def remove_file(large_list,sublist_to_remove):
    # Find the index where the sublist to remove starts
    start_index = large_list.index(sublist_to_remove[0])
    # Remove the sublist from the large list
    large_list = large_list[:start_index] + large_list[start_index + len(sublist_to_remove):]
    return large_list


def cut_df(df,target_cum_sum):
    """cut df rows until it has reached the target value"""

    # Calculate cumulative sum
    cum_sum = df['num_tokens'].cumsum()
    # Find the index where the cumulative sum exceeds or equals the target value
    index_to_cut = cum_sum[cum_sum >= target_cum_sum].index.min()
    # If no index found, keep all rows
    if pd.isnull(index_to_cut):
        index_to_cut = len(df)
    # Remove rows after the index_to_cut
    df = df.iloc[:index_to_cut]
    return df


def get_oov_mat(filename_path:str, train_freq_dir:str, text_dir:str,out_dir:str):

    """
    constructhe target pseudo dataset to estimate oov token freq
    """
    def count_token(text):
        return len(text.split())

    # read train filename
    file_lst = pd.read_csv(filename_path,header=None)[0].tolist()
    # get the list of the candidate txt file
    all_file_lst = os.listdir(text_dir)
    candi_lst = remove_file(all_file_lst,file_lst)
    # loop train_freq file
    for file in tqdm(os.listdir(train_freq_dir)):
        # count token numbers
        train_num = pd.read_csv(train_freq_dir + file)['Freq'].sum()
        oov_sum = 0
        train_frame = pd.DataFrame()
        # get constructed set
        n = 0
        while oov_sum < train_num:
            txt = candi_lst[n]
            frame = txt2csv(text_dir, txt)
            oov_sum += frame['num_tokens'].sum()
            train_frame = pd.concat([train_frame,frame])
            n += 1
        # cut additional line to align with the target train set
        train_frame = cut_df(train_frame,train_num)
        # print out the utt
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        train_frame.to_csv(out_dir + file)


get_oov_mat(filename_path, train_freq_dir, text_dir,out_dir)



