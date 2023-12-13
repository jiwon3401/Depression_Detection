import glob
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline
    
train_list = "/home/ncp/workspace/blocks1/nasw_bak_20231018/utils/train_split.txt"
val_list = "/home/ncp/workspace/blocks1/nasw_bak_20231018/utils/validation_split.txt"
test_list = "/home/ncp/workspace/blocks1/nasw_bak_20231018/utils/test_split.txt"


def get_label(data_path):
    lines = open(data_path).read().splitlines()
    data_label, data_list = [], []
    for index, line in enumerate(lines):
        speaker_label = int(line.split('-')[1])
        file_name     = line.split('-')[0]
        data_label.append(speaker_label)
        data_list.append(file_name)
    return data_label, data_list
    

train_data_label, train_data_list = get_label(train_list)
val_data_label, val_data_list = get_label(val_list)
test_data_label, test_data_list = get_label(test_list)

validation_list = val_data_label + test_data_label
len(validation_list)


train_depression_no, train_depression_yes = [num for num in train_data_label if num<11], [num for num in train_data_label if num>=11]
print(f'not depression: {len(train_depression_no)}')
print(f'depression: {len(train_depression_yes)}')


validation_list = val_data_label + test_data_label
val_depression_no, val_depression_yes = [num for num in validation_list if num<11], [num for num in validation_list if num>=11]
print(f'not depression: {len(val_depression_no)}')
print(f'depression: {len(val_depression_yes)}')

fig, ((ax1, ax2),(ax3,ax4),(ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(20,10))
sns.distplot(train_depression_no, ax=ax1)
sns.distplot(train_depression_yes, ax=ax2)
sns.distplot(val_depression_no, ax=ax3)
sns.distplot(val_depression_yes, ax=ax4)
sns.distplot(train_data_label, ax=ax5)
sns.distplot(validation_list, ax=ax6)

ax1.set(title='train data no depression')
ax2.set(title='train data depression')
ax3.set(title='test data no depression')
ax4.set(title='test data depression')
ax5.set(title='train data distribution')
ax6.set(title='test data distribution')
plt.show()