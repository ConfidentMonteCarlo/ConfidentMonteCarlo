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
import statistics

parser = argparse.ArgumentParser(description='Transformer sampling')
parser.add_argument('--modelpath', type=str, default='./mymodel', help='path of the model')
parser.add_argument('--testpath', type=str, help='password test set')
parser.add_argument('--savepath', type=str, default='./transformer_guess.txt', help='path of the sample file')

parser.add_argument('--n_samples', type=int, default=206848, help='number of samples')
parser.add_argument('--n_samples2', type=int, default=5120, help='number of samples in deriving lb2')
parser.add_argument('--n_repetitions', type=int, default=186, help='number of repetitions')
parser.add_argument('--epsilon', type=float, default=0.005, help='epsilon in Thm 1')
parser.add_argument('--delta', type=float,default=0.333, help='delta in Thm 2')
args = parser.parse_args()

#read model and test dataset
model_eval = transformer_inference(args.modelpath)
df_test = read_file(args.testpath, 'dataset')
df_test['pw'] = df_test['pw'].astype(str)
df_test['freq'] = df_test['freq'].astype(int)
total = df_test['freq'].sum()
print(f'test total: {total}')

#generating guessing numbers for lambda_lb1 and lambda_ub1
print(args.n_samples)
samples = model_eval.generate_samples_in_parallel(args.n_samples)
df_samples = pd.DataFrame(samples, columns=['pwd', 'logprob'])
df_samples.sort_values(by=['logprob'], ascending=False, inplace=True)
estimator = model.PosEstimator(df_samples['logprob'].to_numpy())

df_test['logprob'] = model_eval.logprob_batch(df_test['pw'].tolist())
df_test.sort_values(by=['logprob'], inplace=True) 
df_test['guessing_num_in'] = df_test['logprob'].apply(estimator.position, args=(True,))
df_test['guessing_num_ex'] = df_test['logprob'].apply(estimator.position, args=(False,))

#generating guessing numbers for lambda_ub2
estimators = []
for i in range(args.n_repetitions):
    samples = model_eval.generate_samples_in_parallel(args.n_samples2)
    df_samples = pd.DataFrame(samples, columns=['pwd', 'logprob'])
    df_samples.sort_values(by=['logprob'], ascending=False, inplace=True)
    estimator = model.PosEstimator(df_samples['logprob'].to_numpy())
    estimators.append(estimator)

def median_estimate(logprob, inclusive):
    return statistics.median([estimators[idx].position(logprob, inclusive) for idx in range(args.n_repetitions)])

df_test['guessing_num_ex_med'] = df_test['logprob'].apply(median_estimate, args=(False,))
df_test = compute_bounds(df_test, args.epsilon, args.delta)
write_file(args.savepath, 'guess', df_test)