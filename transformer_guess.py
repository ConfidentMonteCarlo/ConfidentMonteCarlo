import os
import argparse
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import csv
import data_preparation
import model
from architecture import TransformerModel, PositionalEncoding, transformer_inference
from globals import is1class8, is1class16, is3class8
from pwdio import *

parser = argparse.ArgumentParser()
parser.add_argument('--samplepath', type=str, default='./transformer_samples.txt', help='number of samples')
parser.add_argument('--modelpath', type=str, default='./mymodel', help='path of the model')
parser.add_argument('--testpath', type=str, help='password test set')
parser.add_argument('--policy', type=str, default='', help='password composition policy')
parser.add_argument('--inclusiveguessing', type=boolean, default=True, help='password composition policy')
parser.add_argument('--savepath', type=str, default='./transformer_guess.txt', help='saved file')
args = parser.parse_args()
print(f'test path {args.testpath}')
df = read_file(args.samplepath, 'sample')
print(f'sample size {df.shape[0]}')
if args.policy:
    print(f'policy {args.policy}')
    df['1class8'] = df['pw'].apply(is1class8)
    df['1class16'] = df['pw'].apply(is1class16)
    df['3class8'] = df['pw'].apply(is3class8)
    df_pcp = df[df[args.policy]==True]
    print(f"pcp size {df_pcp['logprob'].size}")
    estimator = model.PosEstimator(np.array(df_pcp['logprob'].tolist()), realsize=df.shape[0])
else:
    print('no policy')
    print(df['logprob'].size)
    estimator = model.PosEstimator(np.array(df['logprob'].tolist()))

def get_guessing_number(logprobs):
    ret = []
    print(args.inclusiveguessing)
    for x in logprobs: 
        ret.append(estimator.position(x, args.inclusiveguessing))
    return ret

eval_model = transformer_inference(args.modelpath)
df_test = read_file(args.testpath, 'dataset')
df_test['pw'] = df_test['pw'].astype(str)
df_test['freq'] = df_test['freq'].astype(int)
total = df_test['freq'].sum()
print(f'test total: {total}')

df_test['logprob'] = eval_model.logprob_batch(df_test['pw'].tolist())
df_test['guessing_num'] = get_guessing_number(df_test['logprob'].tolist())
df_test.sort_values(by=['logprob'], inplace=True) 
write_file(args.savepath, 'guess', df_test)