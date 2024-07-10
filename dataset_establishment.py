import copy
import json
import os
import shutil
from tqdm import tqdm
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# folder_path = '../HUST-OBS/deciphered'
folder_path = '/Users/yifangbai/code/Github/7600-baselines/HUST-OBS/deciphered' # modify paths
dataset = []
for root, directories, files in tqdm(os.walk(folder_path)):

    for file in files:
        if'ID'in file or 'DS_Store' in file:
            continue
        data = {}
        file_path = os.path.join(root, file)

        data['label'] = int(file[2:6])
        data['path'] = file_path
        dataset.append(copy.deepcopy(data))

print(f'Total files collected: {len(dataset)}')
# with open('VIT_train.json', 'w', encoding='utf-8') as f:
#     json.dump(dataset, f, ensure_ascii=False)
random.shuffle(dataset)

# 分割数据集为训练集和测试集
test_ratio = 0.2
split_index = int(len(dataset) * (1 - test_ratio))
train_dataset = dataset[:split_index]
test_dataset = dataset[split_index:]

# 将训练集和测试集分别保存到 JSON 文件
with open('VIT_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_dataset, f, ensure_ascii=False)

with open('VIT_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_dataset, f, ensure_ascii=False)

print(f'Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}')