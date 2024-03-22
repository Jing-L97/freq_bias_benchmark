import pandas as pd

def merge_gen(new_dir,old_dir,model):
    """
    merge old and new generations
    input: path to the newly generated tokens and old gen
    return
        df with matched generation
    """

    # loop the new dir
    for file in os.listdir(new_dir):
        gen = pd.read_csv(new_dir + file)
        gen = gen.loc[:, 'filename':]
        # append the result
        gen.pop('LSTM_generated')
        # rename the column
        gen = gen.rename(columns={'LSTM_segmented': 'unprompted_' + temp})
    # append it to the larger generation

    return