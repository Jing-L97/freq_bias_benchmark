"""get and match count between the reference and the comparison corpora"""
import argparse
import os
import sys
import pandas as pd
from lm_benchmark.count_util import count_ngrams

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--root_path', type=str, default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/',
                        help='root path to the utterance and freq dir')

    parser.add_argument('--model', type=str, default='400',
                        help='model name')

    parser.add_argument('--test_set', type=str, default='ind',
                        help='ind,ood or gen')

    parser.add_argument('--ngram', type=int, default=4,
                        help='ngram to extract')

    return parser.parse_args(argv)



def get_score(train_freq, gen_freq):
    '''
    assign scores for freq comparison
    '''
    if gen_freq > train_freq:
        return 1
    elif gen_freq == train_freq:
        return 0
    elif gen_freq < train_freq:
        return -1



def match_ngrams(ref_count,test_count):

    """merge ref and test counts dataframe"""
    merged_count = pd.merge(ref_count,test_count, on='Word', how='outer', suffixes=('_ref', '_test')).fillna(0)
    # add the vocab type
    merged_count['Type'] = 'inv'
    merged_count.loc[merged_count['Count_ref'] == 0, 'Type'] = 'oov'
    merged_count.loc[merged_count['Count_test'] == 0, 'Type'] = 'missing'
    # get the score based on count difference
    merged_count['Score'] = merged_count.apply(lambda row: get_score(row['Count_ref'], row['Count_test']), axis=1)
    return merged_count


def main(argv):
    # load args
    args = parseArgs(argv)
    test_set = args.test_set
    ngram = args.ngram
    ref_path = args.root_path + 'utt/' + args.model + '/gen.csv'
    test_path = args.root_path + 'utt/' + args.model + '/'+ test_set + '.csv'
    out_path = args.root_path + 'freq/' + args.model + '/' + str(ngram) + '_gram/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load ref and test csv
    ref_utt = pd.read_csv(ref_path)['train']
    # count ngrams
    ref_count = count_ngrams(ref_utt, ngram)
    if test_set != 'gen':
        test_utt = pd.read_csv(test_path)['content']
        test_count = count_ngrams(test_utt, ngram)
        merged_count = match_ngrams(ref_count, test_count)
        merged_count.to_csv(out_path + test_set + '.csv')

    # merge the word list
    elif test_set == 'gen':
        # go over the train and gen freq df
        temp_lst = ['0.3','0.6', '1.0', '1.5']
        gen_freq = pd.read_csv(test_path)
        for temp in temp_lst:
            test_utt = gen_freq['unprompted_' + temp]
            test_count = count_ngrams(test_utt, ngram)
            merged_count = match_ngrams(ref_count, test_count)
            merged_count.to_csv(out_path + 'gen_' + temp + '.csv')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)



