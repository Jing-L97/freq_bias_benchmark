"""get scores of the accumulator model"""
import argparse
import os
import sys
import pandas as pd
from lm_benchmark.count_util import count_ngrams
from lm_benchmark.accum_util import *

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--root_path', type=str, default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/',
                        help='root path to the utterance and freq dir')
    parser.add_argument('--model', type=str, default='400',
                        help='model name')
    parser.add_argument('--ngram', type=int, default=5,
                        help='ngram to extract')
    return parser.parse_args(argv)


def get_prob(token_frame,count_header:str,ref_corpus_size:int,gen_corpus_size:int):
    """get prob from the accumulator model"""
    token_count = token_frame[count_header]
    listofscores=[accu_model_tok_stats(wc,ref_corpus_size,gen_corpus_size) for wc in token_count]
    dict_of_dicts = {i: d for i, d in enumerate(listofscores)}
    df = pd.DataFrame.from_dict(dict_of_dicts, orient='index')
    df['freq_per_M'] = token_frame[count_header] / ref_corpus_size * 1000000
    merged_count = pd.concat([token_frame, df], axis=1)
    merged_count = merged_count.rename(columns={'Count': 'Count_ref'})
    merged_count['Type'] = 'inv'
    return merged_count


def main(argv):
    # load args
    args = parseArgs(argv)
    ngram = args.ngram
    ref_path = args.root_path + 'utt/' + args.model + '/gen.csv'
    out_path = args.root_path + 'freq/' + args.model + '/' + str(ngram) + '_gram/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load ref csv
    ref_utt = pd.read_csv(ref_path)['train']
    # count ngrams
    token_frame = count_ngrams(ref_utt, ngram)
    ref_corpus_size = token_frame['Count'].sum()
    merged_count = get_prob(token_frame,'Count',ref_corpus_size,ref_corpus_size)
    merged_count.to_csv(out_path + 'accum.csv')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)



