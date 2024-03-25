"""
compare the freq between the generation and train set
"""
import pandas as pd
import os
from tqdm import tqdm
from lm_benchmark.compare_util import get_freq_table

text_path = '/data/freq_bias_benchmark/data/train/oov/train_utt/'
out_path = '//data/freq_bias_benchmark/data/train/oov/gen_freq/'
column_list = ['train']

def get_freq(text_path:str, out_dir:str,column_list:list):
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
        text = pd.read_csv(text_path + file)
        # loop over different column header
        for col in column_list:
            freq = get_freq_table(text[col].tolist())
            freq.to_csv(out_path + col + '_' + file)

get_freq(text_path, out_path,column_list)


def eval_freq(train_freq, gen_freq):
    '''
    assign scores for freq comparison
    '''
    if train_freq == 0:
        return 'oov'
    else:
        if gen_freq > train_freq:
            return 1
        elif gen_freq == train_freq:
            return 0
        elif gen_freq < train_freq:
            return -1


def match_freq(gen_path:str):
    '''
    map exp and filter freq: loop over the generated tokens
    input: path to train adn gen freq
    return updated gen csv
    '''

    train_freq = pd.read_csv(gen_path + 'train_' + gen_path.split('/')[-2] + '.csv')   # load train freq
    # go over the train and gen freq df
    for file in tqdm(os.listdir(gen_path)):
        if not file.startswith('train'):
            gen_freq = pd.read_csv(gen_path + file)
            # match the freq info
            frame = pd.DataFrame()
            n = 0
            while n < gen_freq.shape[0]:
                gen_row = gen_freq.iloc[[n]]
                try:
                    selected_row = train_freq[train_freq['Word'] == gen_row['Word'].item()]
                    for header in ['Freq', 'Norm_freq', 'Norm_freq_per_million', 'Log_freq', 'Log_norm_freq_per_million']:
                        gen_row['train_' + header] = selected_row[header].item()
                except:
                    # fill zeros to the current dataframe; fill in zeros to the dataframe
                    for header in ['Freq', 'Norm_freq', 'Norm_freq_per_million', 'Log_freq', 'Log_norm_freq_per_million']:
                        gen_row['train_' + header] = 0
                # concatenate the rows
                frame = pd.concat([frame, gen_row])
                n += 1

        # assign comparison score   train_freq, gen_freq
        frame['score'] = frame.apply(lambda row: eval_freq(row['train_Freq'], row['Freq']), axis=1)
        # output the appended freq
        frame.to_csv(gen_path + file)
    return frame


def match_freq(gen_path:str,train_path:str,out_path:str):
    '''
    map exp and filter freq: loop over the generated tokens
    input: path to train adn gen freq
    return updated gen csv
    '''
    train_freq = pd.read_csv(train_path)
    # go over the train and gen freq df
    for file in tqdm(os.listdir(gen_path)):
        if not file.startswith('train'):
            gen_freq = pd.read_csv(gen_path + file)
            # select oov
            gen_freq = gen_freq[gen_freq['score'] == 'oov']
            frame = pd.DataFrame()
            n = 0
            while n < gen_freq.shape[0]:
                gen_row = gen_freq.iloc[[n]]
                try:
                    selected_row = train_freq[train_freq['Word'] == gen_row['Word'].item()]
                    for header in ['Freq', 'Norm_freq', 'Norm_freq_per_million', 'Log_freq', 'Log_norm_freq_per_million']:
                        gen_row['train_' + header] = selected_row[header].item()
                except:
                    # fill zeros to the current dataframe; fill in zeros to the dataframe
                    for header in ['Freq', 'Norm_freq', 'Norm_freq_per_million', 'Log_freq', 'Log_norm_freq_per_million']:
                        gen_row['train_' + header] = 0
                # concatenate the rows
                frame = pd.concat([frame, gen_row])
                n += 1

        # assign comparison score   train_freq, gen_freq
        frame['score'] = frame.apply(lambda row: eval_freq(row['train_Freq'], row['Freq']), axis=1)
        # output the appended freq
        frame.to_csv(out_path + file)
    return frame


gen_path = '/data/freq_bias_benchmark/data/generation/gen_freq/inv/400/'
train_path = '/data/freq_bias_benchmark/data/train/oov/train_freq/400/train_400.csv'
out_path = '/data/freq_bias_benchmark/data/generation/gen_freq/inv/400/'
match_freq(gen_path,train_path,out_path)


def main():

    # get freq of both generated freq and inv freq



    # match freq between train set and gen set


    # match oov tokens between pseudo set and oov parts in the genration