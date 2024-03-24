"""
Construct same register of train data to estimate oov freq
"""

import os
import pandas as pd
from tqdm import tqdm
from lm_benchmark.datasets.parsing_utils.train_parser import clean_text

filename_path = '/data/freq_bias_benchmark/data/train/filename/7100.csv'
train_freq_dir = '/data/freq_bias_benchmark/data/train/train_freq/'
text_dir = '/data/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/'
out_dir = '/data/freq_bias_benchmark/data/train/oov/train_utt/'
def get_oov_mat(filename_path:str, train_freq_dir:str, text_dir:str,out_dir:str):

    """

    """
    def count_token(text):
        return len(text.split())

    # read train filename
    file_lst = pd.read_csv(filename_path,header=None)[0].tolist()

    # loop train_freq file
    for file in tqdm(os.listdir(train_freq_dir)):
        # count token numbers
        train_num = pd.read_csv(train_freq_dir + file)['Freq'].sum()
        oov_sum = 0
        train_frame = pd.DataFrame()
        
        while oov_sum < train_num:
            for txt in tqdm(os.listdir(text_dir)):
                if txt not in file_lst:
                    # get the token num
                    with open(text_dir + txt, encoding="utf8") as f:
                        lines = f.readlines()
                        cleaned_lines = clean_text(lines)
                        frame = pd.DataFrame(cleaned_lines)
                        # assign column headers
                        frame = frame.rename(columns={0: 'train'})
                        frame['num_tokens'] = frame['train'].apply(count_token)
                        frame.insert(loc=0, column='filename', value=file)
                        oov_sum += frame['num_tokens'].sum()
                        train_frame = pd.concat([train_frame,frame])

        # print out the utt
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        train_frame.to_csv(out_dir + file)


get_oov_mat(filename_path, train_freq_dir, text_dir,out_dir)