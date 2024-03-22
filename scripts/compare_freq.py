"""
compare the freq
"""

import pandas as pd
import os
from tqdm import tqdm
from lm_benchmark.compare_util import get_freq_table


"""
analyze the generated tokens
"""

def merge_gen():
    """merge old and new generations"""


def get_freq():
    """
    get freq from the generated tokens
    Returns
    -------
    freq table of the gen
    """

    return
def compare_freq(gen_path, train_path):
    '''
    map exp and filter freq: loop over the generated tokens
    input: generation.csv; freq_table
    return
        - stat.csv for each model and temp
        - freq stat for each model and temp
    '''

    gen_frame = pd.read_csv(gen_path)
    gen_frame['month'] = gen_frame['month'].astype(int)
    # loop over the model list/month
    frame_all = pd.DataFrame()
    for model, age_range in tqdm(month_dict.items()):

        # get gen freq
        gen_frame_selected = gen_frame[(gen_frame['month'] >= age_range[0]) & (gen_frame['month'] <= age_range[1])]

        if not os.path.exists(out_path + '/freq/' + model + '_' + temp + '.csv'):
            gen_freq = get_freq_table(gen_frame_selected['unprompted_' + str(temp)].tolist())
            gen_freq.to_csv(out_path + '/freq/' + model + '_' + temp + '.csv')
        else:
            print('There exists the file ' + model + '_' + temp)
            gen_freq = pd.read_csv(out_path + '/freq/' + model + '_' + temp + '.csv')
        # get train freq
        train_freq = pd.read_csv(train_path + model + '.csv')
        train_freq = train_freq.loc[:, 'Word':]
        # map token freq
        frame = pd.DataFrame()
        # match the freq
        n = 0
        while n < gen_freq.shape[0]:
            gen_row = gen_freq.iloc[[n]]
            try:
                selected_row = train_freq[train_freq['Word'] == gen_row['Word'].tolist()[n]]
            except:
                # fill zeros to the current dataframe; fill in zeros to the dataframe
                selected_row = pd.DataFrame([0, 0, 0, 0, 0, 0]).T
                selected_row.columns = ['Word', 'Freq', 'Norm_freq', 'Norm_freq_per_million', 'Log_freq',
                                        'Log_norm_freq_per_million']

            frame = pd.concat([frame, selected_row])

            n += 1

        frame.to_csv(out_path + model + '_' + temp + '.csv')
        # append model info
        frame['model'] = model
        frame['temp'] = temp
        frame_all = pd.concat([frame_all, frame])

    # output the df
    frame_all.to_csv(out_path + temp + '.csv')
    return frame_all


"""
plot the results for the generatred tokens
"""