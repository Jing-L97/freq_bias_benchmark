import os
import pandas as pd
import enchant
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


root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/freq/400/1_gram/'
out_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/nonwords/'

nonword_dict = {}
temp_lst = ['0.3','0.6','1.0','1.5']
for temp in temp_lst:
    nonword_type,nonword_ratio = check_seg(root_path + 'gen_' + temp + '.csv', 'Count_test', out_dir + temp + '.csv')
    nonword_dict[temp] = nonword_type,nonword_ratio



train_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/freq/400/1_gram/accum.csv'
out_dir = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/nonwords/train.csv'
nonword_type,nonword_ratio = check_seg(train_path, 'Count_ref', out_dir)



def clean_txt(loaded):
    '''
    input: the string list
    return: the cleaned files
    '''
    result = [line for line in loaded if line.strip()]
    all = []
    for sent in tqdm(result):
        # remove punctuations
        #translator = str.maketrans('', '', string.punctuation.replace("'", "") + string.digits)
        translator = str.maketrans('', '', string.punctuation + string.digits)
        translator[ord('-')] = ' '  # Replace hyphen with blank space
        clean_string = sent.translate(translator)
        clean_string = re.sub(r'\s+', ' ', clean_string)
        #clean_string = clean_string.strip().lower()
        # remove additional
        if len(clean_string) > 0:
            all.append(clean_string)
    return all



# read and concatenate files in the
root_path = '/Users/jliu/PycharmProjects/Machine_CDI/Lexical-benchmark_data/train_phoneme/dataset/'
file_lst = pd.read_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/train/inv/filename/400.csv', header=None)
# read and load the files
all_frame = pd.DataFrame()
for file in file_lst[0]:
    # read files
    with open(root_path + file, 'r') as f:
        raw = f.readlines()
        sent = clean_txt(raw)
        utt_frame = pd.DataFrame(sent)
        utt_frame = utt_frame.rename(columns={0: 'train'})
        utt_frame['filename'] =  file
        all_frame = pd.concat([all_frame,utt_frame])

all_frame['num_tokens'] = all_frame['train'].apply(count_words)
all_frame.to_csv('/Users/jliu/PycharmProjects/freq_bias_benchmark/train/inv/train_utt/400_cased_hyphen.csv')

subdfs = np.array_split(all, 10)
root_path = '/train/inv/train_utt/prompt/100/'
# Save each sub-dataframe with the name from 0 to 9
for i, subdf in enumerate(subdfs):
    subdf.to_csv(root_path + f"{i}.csv", index=False)




