"""
match and compare the freq between the generation and train set
"""
import pandas as pd
import os
from tqdm import tqdm
from lm_benchmark.compare_util import get_freq_table


def get_freq(text_path:str, out_dir:str,column_list:list,mode:str):
    """
    get freq from the train/generated tokens
    input:
        path: path to the text path
        column_list: header list for the
    Returns
    -------
    freq table of the gen
    """
    for file in tqdm(os.listdir(text_path)):
        out_path = out_dir + file[:-4] + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if file.endswith('.csv'):
            text = pd.read_csv(text_path + file)
            # loop over different column header
            for col in column_list:
                freq = get_freq_table(text[col].tolist())
                freq.to_csv(out_path + mode + '.csv')



mode = 'ind'
if mode == 'ind':
    column_list = ['train']
elif mode == 'ood':
    column_list = ['content']
text_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/oov/train_utt/' + mode + '/'
out_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/oov/train_freq/'

get_freq(text_path, out_dir,column_list,mode)


# check word num
ind = pd.read_csv(out_dir + 'train_400/ind.csv')['Freq'].sum()
ood = pd.read_csv(out_dir + 'train_400/ood.csv')['Freq'].sum()

train_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_freq/train_400.csv'
train = pd.read_csv(train_dir)['Freq'].sum()