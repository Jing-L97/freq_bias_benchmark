"""
common func to prepare for generation set
"""
from lm_benchmark.datasets.parsing_utils.train_parser import clean_text
import pandas as pd
import numpy as np
def count_token(text):
    return len(text.split())

def txt2csv(text_dir:str, txt:str):
    """convert txt file into csv dataframe: filename\ train\ num_token """
    # read train filename
    with open(text_dir + txt, encoding="utf8") as f:
        lines = f.readlines()
        cleaned_lines = clean_text(lines)
        frame = pd.DataFrame(cleaned_lines)
        # assign column headers
        frame = frame.rename(columns={0: 'train'})
        frame['num_tokens'] = frame['train'].apply(count_token)
        frame.insert(loc=0, column='filename', value=txt)
    return frame

def get_equal_quantity(data_frame, col_header:str, n_bins:int):
    '''
    get bins with same quantity of points
    input: a sorted array or a list of numbers; computes a split of the data into n_bins bins of approximately the same size
    return
        bins: array with each bin boundary
        data_frame: updated df with an additional column of group
    '''
    # sort the dataframe
    data_frame[col_header] = data_frame[col_header].astype(float)
    data_frame = data_frame.sort_values(by=[col_header])
    data = data_frame[col_header]
    # preparing data (adding small jitter to remove ties)
    size = len(data)
    assert n_bins <= size, "too many bins compared to data size"
    mindif = np.min(np.abs(np.diff(np.sort(np.unique(data)))))  # minimum difference between consecutive distinct values
    jitter = mindif * 0.01  # this small jitter will not change the relative order between datapoints
    data_jitter = np.array(data) + np.random.uniform(low=-jitter, high=jitter, size=size)
    data_sorted = np.sort(data_jitter)  # little jitter to remove ties

    # Creating the bins with approx equal number of observations
    bin_indices = np.linspace(1, len(data), n_bins + 1) - 1  # indices to edges in sorted data
    bins = [data_sorted[0]]  # left edge inclusive
    bins = np.append(bins, [(data_sorted[int(b)] + data_sorted[int(b + 1)]) / 2 for b in bin_indices[1:-1]])
    bins = np.append(bins, data_sorted[-1] + jitter)  # this is because the extreme right edge is inclusive in plt.hits

    # computing bin membership for the original data; append bin membership to stat
    bin_membership = np.zeros(size, dtype=int)
    for i in range(0, len(bins) - 1):
        bin_membership[(data_jitter >= bins[i]) & (data_jitter < bins[i + 1])] = i
    data_frame['group'] = bin_membership
    return data_frame



def get_left_border(start, end, k):
    # Initialize a list to store the left border numbers
    borders = []
    # Iterate over the range with fixed interval k and add the left border of each interval to the list
    for i in range(start, end, k):
        borders.append(i)
    return borders



def get_equal_range(data_frame, col_header:str, n_bins:int):
    '''
    get equal-sized bins
    input: a sorted array or a list of numbers; computes a split of the data into n_bins bins of approximately the same size
    return
        bins: array with each bin boundary
        data_frame: updated df with an additional column of group
    '''
    # sort the dataframe
    data_frame[col_header] = data_frame[col_header].astype(float)
    data_frame = data_frame.sort_values(by=[col_header])
    # Determine the range of the column
    min_val = data_frame[col_header].min()
    max_val = data_frame[col_header].max()
    total_range = max_val - min_val
    # get a list of segmented left indices
    bin_range = total_range / n_bins
    # loop over the dataframe to get the group
    data_frame_all = pd.DataFrame()
    n = 0
    while n < n_bins:
        # select subdataframe rows
        if n != n_bins - 1:
            selected_frame = data_frame[data_frame[col_header].between(*[n*bin_range,(n + 1)*bin_range])]
        else:
            selected_frame = data_frame[data_frame[col_header].between(*[n * bin_range, max_val])]
        selected_frame['group'] = n
        data_frame_all = pd.concat([data_frame_all,selected_frame])
        n += 1
    return data_frame_all





# re-calculate bins by same range of each bin
def load_data(freq_path,file,y_header,max_freq,mode, num_bins, oov=False):
    """
    load data to plot figures
    mode: quantity(equal number og points )
    """
    freq_frame = pd.read_csv(freq_path + file)
    # if incorporate oov, convert the score into -1, otherwise remove the oov words
    if oov:
        freq_frame['score'] = freq_frame['score'].replace('oov', 1)
    else:
        # remove oov words
        freq_frame = freq_frame[freq_frame['score']!='oov']
    freq_frame['train_Log_norm_freq_per_million'] = freq_frame['train_Log_norm_freq_per_million'].astype(float)
    freq_frame[y_header] = freq_frame[y_header].astype(float)

    # bin the data to plot trends
    if mode == 'quantity':
        freq_frame = get_equal_quantity(freq_frame,'train_Log_norm_freq_per_million', num_bins)
    elif mode == 'range':
        freq_frame = get_equal_range(freq_frame,'train_Log_norm_freq_per_million', num_bins)

    binned_freq_frame = freq_frame.groupby('group').agg({'train_Log_norm_freq_per_million': 'mean',
                                                      y_header: 'mean'})
    max_freq.append(freq_frame['train_Log_norm_freq_per_million'].max())
    return freq_frame,binned_freq_frame,max_freq





# Create a DataFrame for demonstration
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Gender': ['Female', 'Male', 'Male', 'Male']}
df = pd.DataFrame(data)

# Original DataFrame
print("Original DataFrame:")
print(df)

# Replace 'Male' with 'Man' in the 'Gender' column


# Updated DataFrame
print("\nDataFrame after replacing values:")
print(df)
