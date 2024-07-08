import pandas as pd
pd.options.mode.chained_assignment = None
import random
import torch
import csv
import globals

class Corpus(object):
    def __init__(self, path, maxlen=16, encoding=True, filtering=True):
        print(path)
        df = pd.read_csv(path, sep='\t', header=None, names=['pwd', 'freq'], quoting=csv.QUOTE_NONE)
        df['freq'] = df['freq'].astype(int)
        df['pwd'] = df['pwd'].astype(str)
        self.df = df
        self.maxlen = maxlen

        # filter out pwds that are longer than maxlen or contain non-askii characters
        if filtering:
            df = df[df['pwd'].map(lambda x: len(x) <= self.maxlen)]
            df = df[df['pwd'].map(lambda x : all(c in globals.alphabet for c in x))]

        # convert characters to integers and append 0s to make sure neural network receive input of the same length
        if encoding:
            df['encoding'] = df['pwd'].map(lambda x: [0] + [globals.encode[c] for c in x] + [1]*(self.maxlen + 1 - len(x)))

        data = df['encoding'].tolist()
        # shuffle training data
        random.shuffle(data)
        self.df = df
        self.data = torch.LongTensor(data)
        
        self.all_freqs = self.df['freq'].tolist()
        self.distinct_pwds = self.df['pwd'].tolist()
        self.all_pwds = [item for item, count in zip(self.distinct_pwds, self.all_freqs) for i in range(count)]
        self.frequencyOf = dict(zip(self.distinct_pwds, self.all_freqs))
        self.size = len(self.all_pwds)







