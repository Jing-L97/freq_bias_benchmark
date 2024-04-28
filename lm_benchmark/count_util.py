"""
func to count different chunks
"""
import collections
import spacy
import pandas as pd
from nltk.util import ngrams

nlp = spacy.load("en_core_web_sm")


def count_NER(text):

    """
    count number of NER
    input: text
    return dataframe of NER count
    """
    doc = nlp(text)
    ent_lst = []
    label_lst = []
    for ent in doc.ents:
        ent_lst.append(ent.text)
        label_lst.append(ent.label_)
    # Group by column 'A' and aggregate column 'B' with count
    df = pd.DataFrame([ent_lst, label_lst]).T
    df.columns = ['entity', 'label']
    ent_df = df.groupby(['entity', 'label']).size().reset_index(name='count')
    ent_df['freq'] = ent_df['count']/len(ent_lst)
    return ent_df



def extract_ngrams(words:list, n:int):
    """Generate n-grams from a list of words"""
    n_grams = list(ngrams(words, n))
    # convert tuple into a string
    output = [' '.join(map(str, t)) for t in n_grams]
    return output

def lowercase_text(text):
    try:
        return text.lower()
    except:
        return text

def count_ngrams(col, n:int):
    """count n-grams from a list of words"""
    # preprocess of the utt
    sentences = col.apply(lowercase_text).tolist() # lower the tokens
    # Convert list of sentences into a single list of words
    word_lst = [word for sentence in sentences for word in str(sentence).split()]
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

