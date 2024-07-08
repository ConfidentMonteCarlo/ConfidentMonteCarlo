import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import model
import numpy as np
import string
import globals
import data_preparation
import copy


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Node:
    def __init__(self, *args):
        self.logprob = 0
        self.pwd = ''
        self.pwd_enc = [0]
        self.length = 0
        self.isEOS = False

    def append(self, dis, idx):
        self.pwd_enc.append(idx)
        if idx == 1 and self.isEOS == False:
            self.isEOS = True
            self.logprob = self.logprob - np.log2(dis[idx])

        if self.isEOS == False:
            self.pwd = self.pwd + globals.decode[idx]
            self.logprob = self.logprob - np.log2(dis[idx])
            self.length = len(self.pwd)

    def grow(self, dis):
        np.random.seed()
        idx = 0
        while ((idx == 0) or (idx == 1 and self.length == 0)):
            idx = np.random.choice(len(dis), 1, p=dis)[0]

        self.append(dis, idx)

    def branch(self, dis):
        children = []
        for i in range(len(dis)):
            new = copy.deepcopy(self)
            new.append(dis, i)
            children.append(new)
        return children

    def __lt__(self, other):
        return self.logprob <= other.logprob



class transformer_inference(TransformerModel, model.Model):
    def __init__(self, modelpath, maxlen=16, batch_size=512):
        super(TransformerModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.model = torch.load(modelpath, map_location=torch.device('cpu'))
        else:
            self.model  = torch.load(modelpath).to(self.device)


        self.model.eval()
        self.start = [0]
        self.end = [1]
        self.maxlen = maxlen
        self.batch_size = batch_size
    

    def predict_next(self, pwd):
        x = self.start + [globals.encode[x] for x in pwd] 
        x = torch.LongTensor(x)
        x = torch.unsqueeze(x,0).t().contiguous().to(self.device)
        mask = self.generate_square_subsequent_mask(x.size(0)).to(self.device)
        with torch.no_grad():
            y = self.model(x, mask).view(-1, len(globals.decode))
            activate = nn.Softmax(dim=1)
            y = activate(y)
        return y[-1].cpu().detach().numpy() 
    
    
    def predict_next_most_probable(self, pwd):
        dis = self.predict_next(pwd)
        max_idx = np.argmax(dis)
        ret = globals.decode[max_idx]
        return ret


    def logprob(self, pwd):
        sequence = [globals.encode[x] for x in pwd] + self.end
        res = 0
        for i in range(len(sequence)):
            state = pwd[:i]
            transition = sequence[i]
            res -= math.log2(self.predict_next(state)[transition])
        return res


    def generate(self, maxlen=100):
        pwd = ''
        cur_len = 0
        logprob = 0

        np.random.seed()
        while cur_len <= self.maxlen:
            dis = self.predict_next(pwd)
            idx = np.random.choice(len(dis), 1, p=dis)[0]
            if idx == 0:
                self.generate()
            elif idx == 1:
                logprob -= math.log2(dis[idx])
                break
            else:
                try:
                    pwd += globals.decode[idx]
                    cur_len += 1 
                    logprob -= math.log2(dis[idx])
                except:
                    print(f'error in generate')
                    break
        return pwd, logprob



    def predict_next_batch(self, pwds, format='text'):
        def prepare_data(pwds, format='text'): 
            if format == 'text':   
                data = []  
                for pwd in pwds:
                    pwd_enc = self.start + [globals.encode[x] for x in pwd]
                    data.append(pwd_enc)

                data = np.array(data).T.tolist()
                data = torch.LongTensor(data).to(self.device)
                return data
            elif format == 'encode':
                data = np.array(pwds).T.tolist()
                data = torch.LongTensor(data).to(self.device)
                return data
            else:
                print('error in predict_next_batch')

        data = prepare_data(pwds, format)
        src_mask = self.generate_square_subsequent_mask(data.size(0)).to(self.device)

        with torch.no_grad():
                output = self.model(data, src_mask).view(-1, len(globals.decode))
                activate = nn.Softmax(dim=1)
                y = activate(output)
        return y[-data.size(1):].cpu().detach().numpy() 


    def predict_next_most_probable_batch(self, pwds):
        dis = self.predict_next_batch(pwds)
        max_idx = np.argmax(dis, axis=1)
        ret = [globals.decode[x] for x in max_idx]

        return ret


    def generate_batch(self):
        node_list = []
        for i in range(self.batch_size):
            node_list.append(Node())

        for length in range(self.maxlen):
            samples = [x.pwd_enc for x in node_list]
            distributions = self.predict_next_batch(samples, 'encode')            
            for i, node in enumerate(node_list):
                node.grow(distributions[i])

        return [(x.pwd, x.logprob) for x in node_list]


    # generate multiple batches of samples in parallel
    def generate_parallel(self, num):
        n_batches = (num // self.batch_size) + 1
        print(f'{n_batches} batches')
        ret = []
        for i in range(n_batches):
            ret += self.generate_batch(self.batch_size)

        return ret

        
    # query the log probabilities of pwds in batch
    def logprob_batch(self, pwds, format = 'text'):
        def eachbatch(pwds, format = 'text'):
            node_list = []
            for i in range(len(pwds)):
                node_list.append(Node())

            pwds_enc = []
            if format == 'text':
                maxlen_input = max([len(x) for x in pwds])
                pwds_enc = [globals.encoding(x, maxlen_input) for x in pwds]
            elif format == 'encode':
                pwds_enc = pwds

            for ch_idx in range(1, len(pwds_enc[0])):
                data = [x.pwd_enc[0:ch_idx] for x in node_list]
                dis = self.predict_next_batch(data, 'encode')
                for pw_idx, node in enumerate(node_list):
                    node.append(dis[pw_idx], pwds_enc[pw_idx][ch_idx])
            return [x.logprob for x in node_list]

        ret = []
        iterations = math.ceil(len(pwds) / self.batch_size)
        for i in range(iterations):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(pwds))
            ret += eachbatch(pwds[start: end])

        return ret

