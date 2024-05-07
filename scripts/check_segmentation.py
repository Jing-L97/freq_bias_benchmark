import os
import pandas as pd
import enchant
from lm_benchmark.count_util import count_ngrams
d_uk = enchant.Dict("en_UK")
d_us = enchant.Dict("en_US")


def is_word(word):

    true_word = ["cant", "wont", "dont", "isnt", "its", "im", "hes", "shes", "theyre", "were", "youre", "lets", "wasnt", "werent", "havent", "ill", "youll", "hell", "shell", "well", "theyll", "ive", "youve", "weve", "theyve", "shouldnt", "couldnt", "wouldnt", "mightnt", "mustnt", "thats", "whos", "whats", "wheres", "whens", "whys", "hows", "theres", "heres", "lets", "wholl", "whatll", "whod", "whatd", "whered", "howd", "thatll", "whatre", "therell", "herell"]
    # Function to check if a word is valid
    try:
        if d_uk.check(word) or d_us.check(word) or d_us.check(word.capitalize()) or d_uk.check(word.capitalize()) or word in true_word:
            return True
        else:
            return False
    except:
        return False


def check_seg(freq_dir:str,header:str,out_dir:str):
    """check segmentation correctness"""
    freq = pd.read_csv(freq_dir)
    freq['Correct'] = freq['Word'].apply(is_word)
    # non-word type
    nonword_type = freq[freq['Correct']==False].shape[0]/freq.shape[0]
    nonword_ratio = freq[freq['Correct']==False][header].sum()/freq[header].sum()
    non_words = freq[freq['Correct']==False]
    non_words.to_csv(out_dir)
    return nonword_type,nonword_ratio



root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/train/train_utt/'
nonword_dict = {}
freq_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/seg_check/freq/'
out_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/seg_check/nonwords/new/'
for file in os.listdir(root_path):
    if file.endswith('.csv'):
        ref_utt = pd.read_csv(root_path + file)['train']
        # count ngrams
        ref_count = count_ngrams(ref_utt, 1)
        ref_count.to_csv(freq_path + file)
        # check seg
        nonword_type, nonword_ratio = check_seg(freq_path + file, 'Count',
                                                out_dir + file)
        nonword_dict[file] = nonword_type, nonword_ratio


# change the file structure








