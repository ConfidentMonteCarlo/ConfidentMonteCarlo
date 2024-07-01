import pandas as pd
import math
import csv
import numpy as np

format = {
    'dataset': ['pw','freq'],
    'guess':['pw', 'freq', 'logprob', 'guessing_num_in', 'guessing_num_ex', 'guessing_num_ex_med', 'lb', 'ub'],
    'sample': ['pw', 'logprob'],
    'dictionary': ['pw', 'logprob']
}

def read_file(path, type):
    df = pd.read_csv(path, sep = '\t', header = None, names = format[type], quoting=csv.QUOTE_NONE)
    return df

def write_file(path, type, df):
    df.to_csv(path, columns = format[type], sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

def compute_bounds(df, epsilon = 0.005, delta = 0.333):
    df.sort_values(by = ['logprob'], inplace = True, ignore_index = True)
    df['prob'] = 1/np.power(2, df['logprob'])
    df['error'] = epsilon /  df['prob']
    total_accounts = df['freq'].sum()
    df['cracked_percentage'] = df['freq'].cumsum() / total_accounts
    df['lb1'] = df['guessing_num_ex'] + 1 - df['error']
    df['lb2'] = df['guessing_num_ex_med'] * delta
    df['lb'] = df.apply(lambda x: max(x['lb1'], x['lb2']), axis=1)
    df['ub'] = df['guessing_num_in'] + df['error']
    return df