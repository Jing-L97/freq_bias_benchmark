import os
import pandas as pd
import argparse
import sys
from collections import Counter
from tqdm import tqdm


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Select test sets by freq')

    parser.add_argument('--root_path', type=str, default='/Users/jliu/PycharmProjects/freq_bias_benchmark/data/batch/100_large/',
                        help='root path to the utterance and freq dir')
    return parser.parse_args(argv)

def count_words(lines:list):
    """count the number of words in each batch"""
    # Flatten the nested list
    word_lst = [word for sentence in lines for word in str(sentence).split("|")]
    # Lowercase the characters and count words and characters
    word_lst = [word.lower() for word in word_lst]
    word_counts = Counter(word_lst)
    word_lengths = {word: len(word) for word in word_lst}
    # Create a DataFrame with words, counts, and lengths
    data = {'word': list(word_counts.keys()),
            'count': list(word_counts.values()),
            'word_len': [word_lengths[word] for word in word_counts.keys()]}
    df_counts = pd.DataFrame(data)
    return df_counts


def sort_files(directory):
    # Get list of files in the directory
    files = os.listdir(directory)
    # Extract epoch and batch numbers from file names
    file_info = [(file, int(file.split('_')[0]), int(file.split('_')[1].split('.')[0])) for file in files]
    # Sort files based on epoch and then batch numbers
    sorted_files = sorted(file_info, key=lambda x: (x[1], x[2]))
    # Get sorted file names
    sorted_file_names = [file[0] for file in sorted_files]
    return sorted_file_names


def select_probe_set(root_path):
    # create out_path dir if there is not
    out_path = root_path + 'probe/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # sort the files based on epoch adn checkpoints
    files = sort_files(root_path + 'freq/')

    # Iterate over batches (T-1, T, T+1) recursively
    stat_lst = []
    for i in tqdm(range(1, len(files) - 1)):
        # Load word count CSV files for batches T-1, T, T+1
        csv_files = [os.path.join(root_path + 'freq/', files[j]) for j in range(i - 1, i + 2)]
        dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

        # remove 25% of words with lowest frequency in batch T
        num_selected_words = max(int(len(dfs[1]) * 0.75), 1)
        df_t_sorted = dfs[1].sort_values(by='count', ascending=True)
        selected_words = df_t_sorted.head(num_selected_words)['word'].tolist()
        # Remove words overlapping with batches T-1 and T+1
        selected_words = [word for word in selected_words if word not in dfs[0]['word'] and word not in dfs[1]['word']]
        selected_df = dfs[1][dfs[1]['word'].isin(selected_words)]
        # get stat for the .csv file
        stat_lst.append([files[i],selected_df.shape[0]])
        # Write probe set to file
        selected_df.to_csv(out_path + files[i])
        print(f"Probe set for batch {files[i]} saved to {out_path}")

    stat_df = pd.DataFrame(stat_lst)
    stat_df.columns = ['filename', 'token_type']
    stat_df.to_csv(root_path + 'stat_probe.csv')
    return stat_df


def main(argv):
    # load args
    args = parseArgs(argv)
    # load file
    root_path = args.root_path
    mat_path = root_path + 'mat/'
    freq_path = root_path + 'freq/'
    if not os.path.exists(freq_path):
        os.makedirs(freq_path)
    stat_lst = []
    for file in tqdm(os.listdir(mat_path)):
        if file.endswith('.txt'):
            with open(mat_path + file, 'r') as f:
                lines = f.readlines()
                # count the number of words
                df_counts = count_words(lines)
                df_counts.to_csv(freq_path + file[:-4] + '.csv') # save df
                # get word token and count
                stat_lst.append([file.split('_')[0],file.split('_')[1][:-4],df_counts.shape[0],
                                 df_counts['count'].sum()])

    # assign column headers
    df = pd.DataFrame(stat_lst)
    df.columns = ['epoch', 'batch', 'token_type', 'token_count']
    df.to_csv(root_path + 'stat.csv')
    select_probe_set(root_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)





