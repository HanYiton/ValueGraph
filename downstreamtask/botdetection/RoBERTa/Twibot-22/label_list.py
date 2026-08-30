from cProfile import label
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
import os
os.environ['LD_LIBRARY_PATH'] = 'workshop/gcc-11.3.0/lib64:' + \
                               os.environ.get('LD_LIBRARY_PATH','')
os.environ['LD_PRELOAD']      = 'workshop/gcc-11.3.0/lib64/libstdc++.so.6'
def str_to_int(s):
    if s == 'human':
        return 1
    else:
        return 0

path1 = Path('workshop/workspace/workspace/TwiBot-22/src/RoBERTa/Twibot-22/data')
data = pd.read_csv(path1 / 'label.csv')
label = data['label']
label = torch.tensor(list(map(str_to_int, label)))
torch.save(label, 'workshop/workspace/workspace/TwiBot-22/src/RoBERTa/Twibot-22/data/label_list.pt')

# data_label = {}
# for id in data['id']:
#     data_label[id] = str_to_int(data['label'][data['id'] == id].item())

# path2 = Path('src/RoBERTa/Twibot-22')
# with open(path2 / 'id_list.json', 'r') as f:
#     id_list = json.loads(f.read())

# label_list = []
# for id in id_list:
#     try:
#         label_list.append(data_label[id])
#     except:
#         label_list.append(-1)
# torch.save(torch.tensor(label_list), path2 / 'label_list.pt')


# """
# test part
# """
# data = torch.load('src/RoBERTa/Twibot-20/label_list.pt')
# for i, item in enumerate(data):
#     if item == -1:
#         print(i)
#         break