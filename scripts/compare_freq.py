"""
compare the freq between the generation and train set
"""
import pandas as pd
import os
from tqdm import tqdm
from lm_benchmark.compare_util import get_freq_table
column_header_lst = ['Freq', 'Norm_freq', 'Norm_freq_per_million', 'Log_freq', 'Log_norm_freq_per_million']


#text_path = '/data/freq_bias_benchmark/data/train/inv/train_utt/'   # only for oov generation reference text
text_path = '/data/freq_bias_benchmark/data/generation/generated/merged/'
out_path = '/data/freq_bias_benchmark/data/generation/gen_freq/inv/400/'
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
                    for header in column_header_lst:
                        gen_row['train_' + header] = selected_row[header].item()
                except:
                    # fill zeros to the current dataframe; fill in zeros to the dataframe
                    for header in column_header_lst:
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
train_path = '/data/freq_bias_benchmark/data/generation/gen_freq/inv/400/train_400.csv'
out_path = '/data/freq_bias_benchmark/data/generation/gen_freq/oov/400/'
#match_freq(gen_path,train_path,out_path)

def load_df(path,starting_col):
    """load df by specifying starting column"""
    train_freq = pd.read_csv(path)
    start_index = train_freq.columns.get_loc(starting_col)
    train_freq = train_freq.iloc[:, start_index:]
    return train_freq

# append another column the missing word to the original dataset
train_freq = load_df(train_path,'Word')
# go over the train and gen freq df
for file in tqdm(os.listdir(gen_path)):
        if not file.startswith('train'):
            gen_freq = load_df(gen_path + file,'Word')
            gen_freq['category'] = 'generation'
            # append additional rows of missing words
            missing_words = list(set(train_freq['Word']) - set(gen_freq['Word']))
            missing_frame = train_freq[train_freq['Word'].isin(missing_words)]
            train_freq_col = missing_frame[column_header_lst]
            # add more columns
            for header in column_header_lst:
                missing_frame[header] = 0
            n = 0
            while n <len(column_header_lst):
                missing_frame['train_' + column_header_lst[n]] = train_freq_col[column_header_lst[n]]
                n += 1
            missing_frame['score'] = 0
            missing_frame['category'] = 'train'
            frame_all = pd.concat([gen_freq,missing_frame])
            frame_all.to_csv(gen_path + file)
def main():

    # get freq of both generated freq and inv freq


    # match freq between train set and gen set


    # match oov tokens between pseudo set and oov parts in the genration