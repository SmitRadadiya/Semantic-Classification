import numpy as np
import pandas as pd
import torch
import torchtext
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import Dataset

class MakeData(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

def get_data(path):
    print(f'Dataset path: {path}')
    data = pd.read_excel(path, names=['class', 'review'])
    for i in range(len(data)):
        if data['class'][i] == 'positive':
            data['class'][i] = 2
        elif data['class'][i] == 'neutral':
            data['class'][i] = 1
        else:
            data['class'][i] = 0
    return data


def get_tokenize_data(data,dim):
    vec = torchtext.vocab.GloVe(name='6B', dim = dim) 

    tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+")
    words = []
    for i in data['review']:
        words.append(tokenizer.tokenize(i))

    words_vec = []
    for i in words:
        words_vec.append((sum(vec.get_vecs_by_tokens(i, lower_case_backup=True)))/len(i))

    return words_vec