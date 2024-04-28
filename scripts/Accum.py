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

    parser.add_argument('--ngram', type=int, default=1,
                        help='ngram to extract')

    return parser.parse_args(argv)


def get_prob():

    listofscores=[accu_model_tok_stats(wc,ref_corpus_size,ref_corpus_size) for wc in token_count]
    dict_of_dicts = {i: d for i, d in enumerate(listofscores)}
    df = pd.DataFrame.from_dict(dict_of_dicts, orient='index')
    df['freq_per_M'] = df['token_count'] / 4000000 * 1000000
    merged_count = pd.concat([token_frame, df], axis=1)
    return merged_count


root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/'
model = '400'
ngram = 1
ref_path = root_path + 'utt/' + model + '/gen.csv'
out_path = root_path + 'freq/' + model + '/' + str(ngram) + '_gram/'
if not os.path.exists(out_path):
        os.makedirs(out_path)

# load ref csv
ref_utt = pd.read_csv(ref_path)['train']
# count ngrams
token_frame = count_ngrams(ref_utt, ngram)
token_count = count_ngrams(ref_utt, ngram)['Count']
ref_corpus_size = token_count.sum()

# Display the DataFrame

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
    ref_count = count_ngrams(ref_utt, ngram)['Count']
    merged_count =
    merged_count.to_csv(out_path + test_set + '.csv')




if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)



