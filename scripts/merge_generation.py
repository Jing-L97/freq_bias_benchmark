'''
merge the generated tokens
'''

import pandas as pd

# append the newly generated results to the dataframe
temp_lst = ['0.3','0.6','1.0','1.5']
model = '400'
generation_path = '/data/freq_bias_benchmark/data/generation/generated/new/400h/02/unprompted/sample_random/'
root_path = '/data/freq_bias_benchmark/data/generation/generated/matched/'
out_path = '/data/freq_bias_benchmark/data/generation/generated/merged/'

def append_row(generation_path:str, root_path:str,out_path:str,temp_lst:list,model:str):

    """
    Parameters
    ----------
    generation_path: dir to the newly generated tokens
    root_path: matched generation
    out_path: the out path to save
    """
    def append_column(root_path:str, temp_lst:list):
        # read files recursivelly
        n = 0
        for temp in temp_lst:
            if n == 0:
                # different naming convention
                gen = pd.read_csv(root_path + temp + '.csv')       # load from previous template
                # get the other info from the first temp
                gen = gen.loc[:, 'filename':]
                gen.pop('LSTM_generated')
                # rename the column
                gen = gen.rename(columns={'LSTM_segmented': 'unprompted_' + temp})
            # append the additional column from the dataframe
            else:
                frame = pd.read_csv(root_path + temp + '.csv')
                gen['unprompted_' + temp] = frame['LSTM_segmented'].tolist()
            n += 1
        return gen

    gen_all = append_column(generation_path,temp_lst)
    generation = pd.read_csv(root_path + model + '.csv')
    # align rows
    generation = generation.loc[:, 'filename':]
    generation = pd.concat([generation,gen_all])
    generation.to_csv(out_path + model + '.csv')
    return generation


append_row(generation_path, root_path,out_path,temp_lst,model)

gen = pd.read_csv('/data/freq_bias_benchmark/data/generation/generated/merged/400.csv')['num_tokens'].sum()
train = pd.read_csv('/data/freq_bias_benchmark/data/train/inv/train_utt/400.csv')['num_tokens'].sum()