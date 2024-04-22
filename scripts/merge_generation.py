'''
merge the generated tokens
'''
import os
import pandas as pd
from lm_benchmark.load_data import load_csv
# append the newly generated results to the dataframe
temp_lst = ['0.3','0.6','1.0','1.5']
model = '400'
generation_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/new/400h/02/'
root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/matched/'
out_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/merged/'


def concat_frame(root_path,temp):
    """concatenate frames from root path"""
    frame = pd.DataFrame()
    for file in os.listdir(root_path + temp):
        # different naming convention
        gen = load_csv(root_path + temp + '/' + file, 'filename')  # load from previous template
        frame = pd.concat([frame, gen])
    return frame

def append_row(generation_path:str, root_path:str,out_path:str,temp_lst:list,model:str):

    """
    Parameters
    ----------
    generation_path: dir to the newly generated tokens
    root_path: matched generation
    out_path: the out path to save
    """
    def append_column(root_path:str, temp_lst:list):
        # read files recursively
        n = 0
        for temp in temp_lst:
            if n == 0:
                gen = concat_frame(root_path,temp)
                gen.pop('LSTM_generated')
                # rename the column
                gen = gen.rename(columns={'LSTM_segmented': 'unprompted_' + temp})
            # append the additional column from the dataframe
            else:
                frame = concat_frame(root_path,temp)
                gen['unprompted_' + temp] = frame['LSTM_segmented'].tolist()
            n += 1
        return gen

    gen_all = append_column(generation_path,temp_lst)
    generation = pd.read_csv(root_path + model + '.csv')
    # align rows
    generation = generation.loc[:, 'filename':]
    generation = pd.concat([generation,gen_all])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    generation.to_csv(out_path + model + '.csv')
    return generation


append_row(generation_path, root_path,out_path,temp_lst,model)

gen = pd.read_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/merged/400.csv')['num_tokens'].sum()

train = pd.read_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/400.csv')['num_tokens'].sum()