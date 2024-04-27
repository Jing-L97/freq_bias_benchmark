"""get and match count between the reference and the comparison corpora"""
import argparse
import collections
import sys
import pandas as pd
from nltk.util import ngrams

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--root_path', type=str, default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/',
                        help='root path to the utterance and freq dir')

    parser.add_argument('--model', type=str, default='400',
                        help='model name')

    parser.add_argument('--test_set', type=str, default='ind',
                        help='which type of words to select; recep or exp')

    parser.add_argument('--ngram', type=int, default=1,
                        help='ngram to extract ')

    return parser.parse_args(argv)


def eval_freq(train_freq, gen_freq):
    '''
    assign scores for freq comparison
    '''
    if gen_freq > train_freq:
        return 1
    elif gen_freq == train_freq:
        return 0
    elif gen_freq < train_freq:
        return -1

def extract_ngrams(words:list, n:int):
    """Generate n-grams from a list of words"""
    n_grams = list(ngrams(words, n))
    # convert tuple into a string
    output = [' '.join(map(str, t)) for t in n_grams]
    return output


def count_ngrams(word_lst:list, n:int):
    """count n-grams from a list of words"""
    # extract ngrams
    ngrams = extract_ngrams(word_lst, n)
    # get freq
    frequencyDict = collections.Counter(ngrams)
    freq_lst = list(frequencyDict.values())
    word_lst = list(frequencyDict.keys())
    fre_table = pd.DataFrame([word_lst, freq_lst]).T
    col_Names = ["Word", "Count"]
    fre_table.columns = col_Names
    return fre_table


def match_ngrams(ref_count,test_count):

    """merge ref and test counts dataframe"""

    merged_count = pd.merge(ref_count,test_count, on='Count', how='outer', suffixes=('_ref', '_test')).fillna(0)
    # add the vocab type
    merged_count['Type'] = 'inv'
    merged_count.loc[merged_count['Count_ref'] == 0, 'Overlap'] = 'oov'
    merged_count.loc[merged_count['Count_test'] == 0, 'Overlap'] = 'missing'
    # get the score based on count difference
    merged_count['Score'] = df.apply(lambda row: custom_function(row['Count_ref'], row['Count_test']), axis=1)
    return merged_count



def main(argv):

    # load args
    args = parseArgs(argv)
    test_set = args.test_set
    ref_path = args.root_path + 'gen.csv'
    test_path = args.root_path + test_set
    ngram = args.ngram

    # load ref and test csv
    ref_utt = pd.read_csv(ref_path)['train']
    # count ngrams
    ref_count = count_ngrams(ref_utt, ngram)
    if test_set != 'gen':
        test_utt = pd.read_csv(ref_path)['content']
        test_count = count_ngrams(test_utt, ngram)
        match_ngrams(ref_count, test_count)

    # merge the word list
    elif test_set != 'gen':
        # go over the train and gen freq df
        temp_lst = ['0.3', '0.6', '1.0', '1.5']
        gen_freq = pd.read_csv(test_path)
        for temp in temp_lst:
            test_utt = gen_freq['unprompted_' + temp]
            test_count = count_ngrams(test_utt, ngram)
            match_ngrams(ref_count, test_count)





if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)



