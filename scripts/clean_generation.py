'''
check an drecycle CHILDES genrations
'''

import pandas as pd


del_lst = [1,2,3,4]
model_path = '/data/exp/unprompted_0.6/'
ordered_path = '/data/exp/ordered/'
def order_column(model_path,ordered_path,model_lst):

    '''
    reorder the temp = 0.3 column
    Parameters
    ----------
    model_path
    ordered_path: output path to the result
    model_lst: target model to match

    Returns
    -------
    the matched dataframe
    '''
    def count_token(text):
        return len(text.split())
    def cut_utt(text,n):
        return ' '.join(text.split()[:int(n)])
    # merge the results
    frame_all = pd.DataFrame()
    for model in model_lst:
        # count the number of tokens in 0.3 temp
        trans = pd.read_csv(model_path + str(model) + '.csv')
        ordered_frame = trans.sort_values(by='num_tokens', ascending=True)
        ordered_frame['len'] = ordered_frame['unprompted_0.3'].apply(count_token)
        ordered_target = ordered_frame.sort_values(by='len', ascending=True)
        ordered_frame['unprompted_0.3'] = ordered_target['unprompted_0.3'].tolist()
        # cut the string into target length
        utt_lst = ordered_frame.apply(
            lambda row: cut_utt(row['unprompted_0.3'], row['num_tokens']), axis=1)
        ordered_frame['unprompted_0.3'] = utt_lst
        ordered_frame.pop('len')
        # append the 0.6 column
        ordered_frame.pop('LSTM_generated')
        # rename the column
        ordered_frame = ordered_frame.rename(columns={'LSTM_segmented': 'unprompted_0.6'})
        last_col_name = ordered_frame.columns[-1]
        position = len(ordered_frame.columns) - 3
        last_col = ordered_frame.pop(last_col_name)
        ordered_frame.insert(position, last_col_name, last_col)
        gen = ordered_frame.loc[:, 'speaker':]
        frame_all = pd.concat([frame_all,gen])
        # print out the result
        ordered_frame.to_csv(ordered_path + str(model) + '.csv')
    frame_all.to_csv(ordered_path + 'generation.csv')



# append the newly generated results to the dataframe
root_path = '/data/exp/generation/'
temp_lst = ['0.3','0.6','1.0','1.5']
model_lst = ['4500h','7100h']
generation_path = '/data/exp/ordered/generation.csv'
out_path = '/data/exp/'
def append_row(generation_path, root_path,out_path,temp_lst,model_lst):
    def append_column(root_path, temp_lst):

        # read files recursivelly
        n = 0
        for temp in temp_lst:
            if n == 0:
                gen = pd.read_csv(root_path + '1_' + temp + '_new.csv')
                # get the other info from the first temp
                gen = gen.loc[:, 'speaker':]
                gen.pop('LSTM_generated')
                # rename the column
                gen = gen.rename(columns={'LSTM_segmented': 'unprompted_' + temp})

            # append the additional column from the dataframe
            else:
                frame = pd.read_csv(root_path + '1_' + temp + '_new.csv')
                gen['unprompted_' + temp] = frame['LSTM_segmented'].tolist()
            n += 1
        return gen

    gen_all = pd.DataFrame()
    for model in model_lst:
        model_path = root_path + model + '/00/unprompted/sample_random/'
        gen= append_column(model_path,temp_lst)
        gen_all = pd.concat([gen_all,gen])
    generation = pd.read_csv(generation_path)
    # align rows
    generation = generation.loc[:, 'speaker':]
    generation = pd.concat([generation,gen_all])
    # align lang notation
    generation['lang'] = generation['lang'].replace('UK', 'BE')
    generation['lang'] = generation['lang'].replace('AME', 'AE')
    generation['lang'] = generation['lang'].fillna('AE')
    generation.to_csv(out_path + 'generation.csv')
    return generation

# align the new and old generations
generation = pd.read_csv(out_path + 'generation.csv')
generation_old = pd.read_csv(out_path + 'generation_old.csv')
gen_lst = generation['lang'].unique()
