import pandas as pd
import numpy as np
import random
import os

path = './dataset'

if not os.path.isdir(path):
    os.mkdir(path)

'''
df_1 = pd.read_json("train_data_prefect_v1.json", orient='index')

df_2 = pd.read_json("train_data_prefect.json", orient='index')

df = pd.concat([df_1,df_2])
'''
df = pd.read_json("train_data_prefect.json", orient='index')

length_tracker={}
for idx, summary in enumerate(df['summary_mask']):

    length = len(summary)
    if length not in length_tracker.keys():
        length_tracker[length] = list()
    length_tracker[length].append(idx)
    
    
distribution = 0.9
train_dataset, valid_dataset = [],[]
for length in length_tracker.keys():
    
    indexes = length_tracker[length]
    random.shuffle(indexes)
    
    train_length = int(len(indexes)//(1/distribution))
    
    train = indexes[:train_length]
    valid = indexes[train_length:]
    
    train_dataset.extend(train)
    valid_dataset.extend(valid)


train_dataset = df.iloc[train_dataset].reset_index(drop=True)
valid_dataset = df.iloc[valid_dataset].reset_index(drop=True)

train_dataset.to_json("./dataset/train.json")
valid_dataset.to_json("./dataset/valid.json")
