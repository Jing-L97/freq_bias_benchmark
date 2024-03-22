#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare generation data
@author: jliu
"""
import argparse
import os
import pandas as pd
import sys
from tqdm import tqdm
from lm_benchmark.datasets.parsing_utils.train_parser import clean_text

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    return parser.parse_args(argv)


#TODO: check dataset loader class
filename_path = '/data/freq_bias_benchmark/data/train/filename/'
text_dir = '/data/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/'
out_dir = '/data/freq_bias_benchmark/data/train/train_utt/'
generation_path = '/data/freq_bias_benchmark/data/generation/generated/generation_old.csv'

month_dict = {'400':[4,8],'800':[9,18],'1600':[19,28],'3200':[29,36],'4500':[46,54],'7100':[66,74]}
temp_lst = ['0.3','0.6','1.0','1.5']
utt_path = '/data/freq_bias_benchmark/data/train/train_utt/'
model_lst = ['800','1600','3200','4500','7100']


def get_train(filename_path:str, text_dir:str,out_dir:str,chunk:str):

    """
    get the train data from the filename
    input:
        filename_path: storing filenames of the target chunk
        text_dir: all the raw .txt
        chunk: the corresponding chunk, influence the filename, raw text
    Returns
        dataframe with columns: filename, train, num_tokens
    """
    def count_token(text):
        return len(text.split())

    # read the filename frame
    file_frame = pd.read_csv(filename_path + chunk + '.csv',header=None)
    train_frame = pd.DataFrame()
    for file in file_frame[0]:
        with open(text_dir + file, encoding="utf8") as f:
            lines = f.readlines()
            cleaned_lines = clean_text(lines)
            frame = pd.DataFrame(cleaned_lines)
            # assign column headers
            frame = frame.rename(columns={0:'train'})
            frame['num_tokens'] = frame['train'].apply(count_token)
            frame.insert(loc=0,column='filename',value=file)
        train_frame = pd.concat([train_frame,frame])

    # save the result
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_frame.to_csv(out_dir + chunk + '.csv')
    return train_frame



def match_generation(utt_path:str,generation_path:str,chunk:str,month_dict:dict
                     ,temp_lst,prompt_type = 'unprompted'):
    
    '''
    segment the train dataset by utterances
    input:
        utt_path: path to the train utt .csv
        generation_path:path to the generated tokens to be matched
    Returns
    -------
        train: .csv train\num_token\generation(prompt_temp)
    '''

    gen_frame = pd.read_csv(generation_path)
    train_frame = pd.read_csv(utt_path + chunk + '.csv')
    # select the generations from the corresponding model
    age_range = month_dict[chunk]
    generation = gen_frame[(gen_frame['month'] >= age_range[0]) & (gen_frame['month'] <= age_range[1])]

    # match by the utterance length
    unmatched_frame = pd.DataFrame()
    matched_frame = pd.DataFrame()
    rest_gen = pd.DataFrame()

    # loop the generated columns to match the train frame
    generation_grouped = generation.groupby('num_tokens')
    for token_num, generation_group in tqdm(generation_grouped):
        # match the utterances with similar tokens
        candi_frame = train_frame[train_frame['num_tokens'] == token_num]
        row_len = min(candi_frame.shape[0], generation_group.shape[0])
        matched_train = candi_frame.head(row_len)
        matched_rows = generation_group.head(row_len)

        # collect unmatched generated rows
        if candi_frame.shape[0] < generation_group.shape[0]:
            rest_rows = generation_group[row_len:]
            rest_gen = pd.concat([rest_gen, rest_rows])

        # collect rest of the train frame
        if candi_frame.shape[0] > generation_group.shape[0]:
            rest_rows = candi_frame[row_len:]
            unmatched_frame = pd.concat([unmatched_frame, rest_rows])

        # append the generation columns to the train dataframe
        for temp in temp_lst:
            matched_train[prompt_type + '_' + temp] = matched_rows[prompt_type + '_' + temp].tolist()
        matched_frame = pd.concat([matched_frame, matched_train])

    # save the result
    matched_frame.to_csv('/'.join(generation_path.split('/')[:-1]) + '/matched/' + chunk + '.csv')
    rest_gen.to_csv('/'.join(generation_path.split('/')[:-1]) + '/rest/' + chunk + '.csv')
    unmatched_frame.to_csv('/'.join(generation_path.split('/')[:-2]) + '/' + chunk + '.csv')
    return matched_frame, rest_gen, unmatched_frame

def segment_generation(utt_path, model_lst, n):
        '''
        segment generation into n subdataframes
        input:
            utt_path: path to the train utt .csv
            n: # subdataframe to be segmented
            model_lst: the list of model generation to be segmented
        Returns
        -------
            train: .csv train\num_token\generation(prompt_temp)
        '''

        def segment_dataframe(df, n):
            num_rows = len(df)
            segment_size = num_rows // n
            remainder = num_rows % n
            segments = []
            start = 0
            for i in range(n):
                if i < remainder:
                    end = start + segment_size + 1
                else:
                    end = start + segment_size
                segments.append(df.iloc[start:end])
                start = end
            return segments

        for model in tqdm(model_lst):
            script = pd.read_csv(utt_path + model + '.csv')
            # Segment the DataFrame
            sub_dataframes = segment_dataframe(script, n)
            n = 0
            for _, sub_df in enumerate(sub_dataframes):
                # save the result
                out_dir = utt_path + 'prompt/' + model + '/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                sub_df.to_csv(out_dir + str(n) + '.csv')
                n += 1


def main(argv):
    # Args parser
    args = parseArgs(argv)
    chunk_lst = ['400','800','1600','3200','4500']
    for chunk in chunk_lst:
        # load/get train frame
        #get_train(filename_path, text_dir, out_dir, chunk)
        # gene
        matched_frame, rest_gen, unmatched_frame = match_generation(utt_path, generation_path, chunk, month_dict
        ,temp_lst)



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
