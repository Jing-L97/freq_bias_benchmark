import pandas as pd
import enchant
d = enchant.Dict("en_UK")

def is_word(word):
    # Function to check if a word is valid
    try:
        return d.check(word)
    except:
        return False



freq = pd.read_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/train/inv/train_freq/400_cased_hyphen.csv')
all_frame = pd.read_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/train/inv/train_utt/400_cased_hyphen.csv')
freq = count_ngrams(all_frame['train'], 1)

freq['Correct'] = freq['Word'].apply(is_word)
freq.to_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/train/inv/train_freq/400_annotated.csv')
# non-word type
nonword_type = freq[freq['Correct']==False].shape[0]/freq.shape[0]
nonword_ratio = freq[freq['Correct']==False]['Count'].sum()/freq['Count'].sum()


non_words = freq[freq['Correct']==False]
non_words.to_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/train/inv/train_freq/400_non.csv')
# non-word type