import pandas as pd

batch_lst = ['100_large/','100_med/','100_small/','400_large/','400_med/','1600_large/','1600_med/']
root_path = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/batch/'

stat_all = []
for batch in batch_lst:
    stat = pd.read_csv(root_path + batch +'/stat.csv')
    stat_probe = pd.read_csv(root_path + batch + '/stat_probe.csv')
    avg_type = stat['token_type'].sum()/stat['token_type'].shape[0]
    avg_token = stat['token_count'].sum()/stat['token_count'].shape[0]
    probe_type = stat_probe['token_type'].sum()/stat_probe['token_type'].shape[0]
    stat_all.append([batch,avg_token,avg_type,probe_type])

stat_frame = pd.DataFrame(stat_all)
print(stat_frame)




hour_lst = ['100','400','1600']
freq_root = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/seg_check/freq/'
nonword_root = '/Users/jliu/PycharmProjects/freq_bias_benchmark/data/seg_check/nonwords/new/'
stat_all = []
for hour in hour_lst:
    freq = pd.read_csv(freq_root + hour + '.csv')
    nonword = pd.read_csv(nonword_root + hour + '.csv')
    type_ratio = nonword.shape[0]/freq.shape[0]
    count_ratio = nonword['Count'].sum()/freq['Count'].sum()
    stat_all.append([hour,type_ratio,count_ratio])

print(stat_all)