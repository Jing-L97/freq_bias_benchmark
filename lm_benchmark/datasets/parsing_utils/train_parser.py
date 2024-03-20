import string
import re
from tqdm import tqdm

def clean_text(loaded:list):
    """
    remove digits and punct of a text string
    Returns
    -------
    a list of the cleaned string
    """
    result = [line for line in loaded if line.strip()]
    cleaned_text = []
    for sent in tqdm(result):
        # remove punctuations
        translator = str.maketrans('', '', string.punctuation + string.digits)
        clean_string = sent.translate(translator)
        clean_string = re.sub(r'\s+', ' ', clean_string)
        clean_string = clean_string.strip()
        cleaned_text.append(clean_string)
    return cleaned_text
