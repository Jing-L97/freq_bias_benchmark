"""
detect n-gram in the generated tokens
"""

from nltk.util import ngrams
import pandas as pd
import os
from tqdm import tqdm
from lm_benchmark.compare_util import get_freq_table,get_ngram_freq

def extract_ngrams(words:list, n:int):
    """Generate n-grams from a list of words"""
    n_grams = list(ngrams(words, n))
    # convert tuple into a string
    output = [' '.join(map(str, t)) for t in n_grams]
    return output


# convert the synthesized data into word lists
text_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/merged/'
out_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/generation/gen_freq/inv/'
column_list = ['train','unprompted_0.3','unprompted_0.6','unprompted_1.0','unprompted_1.5']

def get_freq(text_path:str, out_dir:str,column_list:list,n_gram:int):
    """
    get freq from the train/generated tokens
    input:
        path: path to the text path
        column_list: header list for the
    Returns
    -------
    freq table of the gen
    """

    def lowercase_text(text):
        try:
            return text.lower()
        except:
            return text

    for file in tqdm(os.listdir(text_path)):
        out_path = out_dir + file[:-4] + '/' + str(n_gram) + '_gram/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if file.endswith('.csv'):
            text = pd.read_csv(text_path + file)
            # loop over different column header
            for col in column_list:
                # lower the tokens
                sentences = text[col].apply(lowercase_text).tolist()
                # Convert list of sentences into a single list of words
                gen = [word for sentence in sentences for word in str(sentence).split()]
                text_lst = extract_ngrams(gen, n_gram)
                freq = get_ngram_freq(text_lst,True)   # problem here!
                freq.to_csv(out_path + col + '_' + file)
n_gram_lst = [3,4,5]
for n_gram in n_gram_lst:
    get_freq(text_path, out_dir,column_list,n_gram)

