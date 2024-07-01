import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import TransformerModel, PositionalEncoding
import data_preparation
import globals

parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--filename', type=str, default='../dataset/rockyou.csv',
                    help='path of the data corpus')
parser.add_argument('--second_training_file', type=str, default='',
                    help='path of the data corpus')                    
parser.add_argument('--embedding_size', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=1024,
                    help='size of hidden layers')
parser.add_argument('--n_layers', type=int, default=16,
                    help='number of Transformer encoder layers')
parser.add_argument('--n_heads', type=int, default=16,
                    help='number of attention heads')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size')
parser.add_argument('--maxlen', type=int, default=16,
                    help='maximum length of passwords')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log_interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--savepath', type=str, default='./',
                    help='path to save the final model')
parser.add_argument('--policy', type=str, default='',
                    help='password policy')
parser.add_argument('--cont', type=bool, default=False,
                    help='load a model and continue to train')
args = parser.parse_args()


print('if continuous training {}'.format(args.cont))
print('torch version {}'.format(torch.__version__))
print('device name {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
print('is cuda available {}'.format(torch.cuda.is_available()))
print('embedding {}, hidden {}, batch {}, n_layers {}'.format(args.embedding_size, args.hidden_size, args.batch_size, args.n_layers))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


corpus = data_preparation.Corpus(args.filename, args.maxlen)
data = corpus.data
if args.second_training_file:
    corpus_sec = data_preparation.Corpus(args.second_training_file, args.maxlen)
    data_sec = corpus_sec.data
    data = torch.cat((data, data_sec),0)
print(data.shape)
train_data = batchify(data, args.batch_size)
print(train_data.shape)

bptt = args.maxlen + 1
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

ntokens = len(globals.decode) # the size of vocabulary
if not args.cont:
    model = TransformerModel(ntokens, args.embedding_size, args.n_heads, args.hidden_size, args.n_layers, args.dropout).to(device)
else:
    model = torch.load('mymodel', map_location=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt + 1)):
        data, targets = get_batch(train_data, i)
        if i < 3:
            print(data)
            print(targets.view(-1,args.batch_size))            
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        log_interval = args.log_interval
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches |'
                  ' ms/batch {:5.2f} |'
                  ' loss {:5.2f} | ppl {:8.2f} '.format(
                    epoch, batch, len(train_data) // bptt,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train()

    print('-' * 90)
    print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
    print('-' * 90)
    
torch.save(model, args.savepath)


