import pandas as pd
import numpy as np
import tqdm
import os
import random

r = 0.6
window = int(r * 2563)  # 2563 is the average full length

valid_set_rate = 0.1
test_set_rate = 0.1
sampling_num = 2e4 / r  # just empirical setting (shorter segments need more sampling number)
k = 20

eps = 1e-2

path = './data/raw_data/'
train_set_path = './data/train_data'
valid_set_path = './data/valid_data'
test_set_path = './data/test_data'
if os.path.exists(train_set_path): os.remove(train_set_path)
if os.path.exists(valid_set_path): os.remove(valid_set_path)
if os.path.exists(test_set_path): os.remove(test_set_path)
file_name_list = os.listdir(path)
random.seed(100)
random.shuffle(file_name_list)

file_len = len(file_name_list)
train_name_list = file_name_list[int(file_len * (test_set_rate + valid_set_rate)) : ]
valid_name_list = file_name_list[: int(file_len * valid_set_rate)]
test_name_list = file_name_list[int(file_len * valid_set_rate) : int(file_len * (test_set_rate + valid_set_rate))]

def sampling(name_list, set_path, sampling_num, resolution):
    dic = dict()
    # build a dict for imbalanced sampling
    for file_name in tqdm.tqdm(name_list):
        df = pd.read_csv(path + file_name)
        key = int((df.iloc[0, 0] - 12.5139) / resolution) * resolution
        if key in dic.keys():
            dic[key].append(file_name)
        else:
            dic[key] = [file_name]

    # imbalanced sampling
    key_num = len(dic.keys())
    sampling_num_per_key = int(sampling_num / key_num) + 1

    def save(data, path):
        with open(path, 'a') as f:
            f.write(','.join(map(str, data)))
            f.write('\n')

    for key, value in tqdm.tqdm(dic.items()):
        sampling_num_per_df = int(sampling_num_per_key / len(value)) + 1
        for file_name in value:
            df = pd.read_csv(path + file_name)
            sampling_interval = df.shape[0] - window
            for i in range(sampling_num_per_df):
                head = int(sampling_interval * i / sampling_num_per_df)
                tail = int(sampling_interval * (i + 1) / sampling_num_per_df)
                start_idx = random.randint(head, tail) if window < df.shape[0] else 0
                end_idx = start_idx + window if window < df.shape[0] else df.shape[0]

                pad_num = window - df.shape[0]
                max_capacity = df.iloc[0, 0]
                voltage = df.iloc[start_idx:end_idx, 1].to_numpy()
                if window > df.shape[0]: voltage = np.pad(voltage, pad_width=(0, pad_num), mode='constant', constant_values=voltage[-1])
                capacity = df.iloc[start_idx:end_idx, 2].to_numpy()
                if window > df.shape[0]: capacity = np.pad(capacity, pad_width=(0, pad_num), mode='constant', constant_values=capacity[-1])
                capacity -= capacity[0]

                sample = voltage
                sample = np.append(sample, capacity)
                sample = np.append(sample, max_capacity)
                sample = sample.tolist()

                save(sample, set_path)

sampling(train_name_list, train_set_path, int(sampling_num * (1 - test_set_rate - valid_set_rate)), 11.47 / k)
sampling(valid_name_list, valid_set_path, int(sampling_num * valid_set_rate), 11.47 / k)
sampling(test_name_list, test_set_path, int(sampling_num * test_set_rate), 11.47 / k)