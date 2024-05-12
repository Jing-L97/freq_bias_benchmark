"""select the unique words from each batch"""
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm


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



# load file
root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/batch/100/'
mat_path = root_path + 'mat/'
freq_path = root_path + 'freq/'
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



directory = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/batch/100/freq/'
def select_probe_set(directory):
    # sort the files based on epoch adn checkpoints
    files = sorted(os.listdir(directory))

    # Iterate over batches (T-1, T, T+1) recursively
    for i in range(1, len(files) - 1):
        # Load word count CSV files for batches T-1, T, T+1
        csv_files = [os.path.join(directory, files[j]) for j in range(i - 1, i + 2)]
        dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

        # Merge word counts for batch T
        df_t = dfs[1]
        for df in dfs[:1] + dfs[2:]:
            df_t = pd.merge(df_t, df, on='word', how='outer')

        # Select 25% of words with lowest frequency in batch T
        num_selected_words = max(int(len(df_t) * 0.25), 1)
        df_t_sorted = df_t.sort_values(by='count', ascending=True)
        selected_words = df_t_sorted.head(num_selected_words)['word'].tolist()

        # Remove words overlapping with batches T-1 and T+1
        for df in dfs[:1] + dfs[2:]:
            selected_words = [word for word in selected_words if word not in df['word'].tolist()]

        # Write probe set to file
        probe_set_path = os.path.join(directory, f'probe_set_{i}.txt')
        with open(probe_set_path, 'w') as file:
            file.write('\n'.join(selected_words))

        print(f"Probe set for batch {files[i]} saved to {probe_set_path}")


# Example usage
directory = 'path/to/your/directory'
select_probe_set(directory)
