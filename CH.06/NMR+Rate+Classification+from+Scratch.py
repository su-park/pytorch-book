
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torch torchtext tqdm')


# # SET LIBRARY

# In[1]:


import os, time, sys
from glob import glob
import numpy as np
import pandas as pd
import torch

from torchtext import data, datasets
from torchtext.vocab import GloVe, FastText, CharNGram
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # LOAD DATA

# In[ ]:


#!pwd
#!tar xvzf nmr-100k.tar.gz


# In[2]:


tab = pd.read_csv('/jet/prs/workspace/nmr-100k.csv',delimiter='|')


# In[3]:


# SANITY CHECK
print(len(tab.loc[0,'rviw_modd'].split()))


# In[4]:


tab['label'] = np.where(tab.rviw_rate>7, "pos", np.where(tab.rviw_rate==1, "neg", "neu"))
tab['input'] = tab.rviw_modd


# In[18]:


pd.crosstab(tab.is_train, tab.label, margins=False)#.apply(lambda r: round(r/r.sum(),3), axis=1)


# In[6]:


tab[tab.is_train==True][tab.label.isin(['pos','neg'])].loc[:,['label','input']].to_json('train.json',orient='records',lines=True) #.sample(frac=0.1, replace=False)
tab[tab.is_train!=True][tab.label.isin(['pos','neg'])].loc[:,['label','input']].to_json('valid.json',orient='records',lines=True) #.sample(frac=0.1, replace=False)


# In[7]:


INPUT = data.Field(fix_length=50, batch_first=False)
LABEL = data.Field(sequential=False,)

fields = {'label': ('label', LABEL), 'input': ('input', INPUT)}

train, valid = data.TabularDataset.splits(
    path = '/jet/prs/workspace',
    train = 'train.json',
    test = 'valid.json',
    format = 'json',
    fields = fields
)


# In[8]:


print(vars(train[0]))


# In[9]:


INPUT.build_vocab(train, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(train,)


# In[ ]:


# FOR DEBUGGING ONLY

#print(INPUT.vocab.freqs)
#print(INPUT.vocab.vectors)
#print(INPUT.vocab.stoi)

INPUT = data.Field(fix_length=50, batch_first=False)
LABEL = data.Field(sequential=False,)

train, valid = datasets.IMDB.splits(INPUT, LABEL)

INPUT.build_vocab(train, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(train,)


# In[10]:


g_train, g_valid = data.BucketIterator.splits((train, valid), batch_size=32, device=-1, shuffle=True, sort=False)
g_train.repeat = False
g_valid.repeat = False
dataloader = {'train':g_train, 'valid':g_valid}
dataset_sizes = {'train':len(g_train.dataset),'valid':len(g_valid.dataset)}


# In[11]:


# SANITY CHECK
batch = next(iter(dataloader['train']))
x_input = batch.input.cuda()
y_label = batch.label.cuda()
print(x_input.size())
print(y_label.size())


# In[12]:


class Gph(nn.Module):
    
    def __init__(self, n_vocab, n_hidden, n_label, btch_size=1, nl=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.btch_size = btch_size
        self.nl = nl
        self.embd = nn.Embedding(n_vocab, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, nl)
        self.fcn = nn.Linear(n_hidden, n_label)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x_input):
        bs = x_input.size()[1]
        if btch_size != self.btch_size:
            self.btch_size = btch_size
        embd = self.embd(x_input)
        
        h0 = c0 = Variable(embd.data.new(*(self.nl, self.btch_size, self.n_hidden)).zero_())
        rnn, _ = self.rnn(embd,(h0, c0))
        rnn = rnn[-1]
        fcn = F.dropout(self.fcn(rnn), p=0.8)
        return self.softmax(fcn)    


# # SET PARMS

# In[13]:


n_vocab = len(INPUT.vocab)
n_hidden = 100


# In[14]:


gph = Gph(n_vocab, n_hidden, n_label=3, btch_size=32)    
if torch.cuda.is_available():
    gph = gph.cuda()


# In[15]:


learning_rate = 0.001
optimizer = optim.Adam(gph.parameters(), lr=learning_rate)

#gph.embd.weight.requires_grad = False
#optimizer = optim.SGD([parm for parm in gph.parameters() if parm.requires_grad==True], lr=learning_rate)


# In[16]:


def train_gph(gph, optimizer, dataloader, phase='train', volatile=False):
    if phase == 'train':
        gph.train()
    if phase == 'valid':
        gph.eval()
        volatile = True

    running_loss = 0.0
    running_corrects = 0
    running_counts = 0

    for idx, batch in enumerate(dataloader[phase]):
        if torch.cuda.is_available():
            x_input, y_label = batch.input.cuda(), batch.label.cuda()

        if phase == 'train':
            optimizer.zero_grad()
        y_prob = gph(x_input)
        _, y_pred = torch.max(y_prob, 1)
        loss = F.nll_loss(y_prob, y_label)

        if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(y_pred == y_label.data).item()
        running_counts += len(y_pred)

    epoch_loss = running_loss / running_counts
    epoch_acc = running_corrects / running_counts
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


# In[17]:


# DEBUGGED
# https://discuss.pytorch.org/t/data-iterator-failing-on-dev-set-only/11956/4
train_loss, train_acc = [],[]
valid_loss, valid_acc = [],[]
for epoch in range(1,16):
    epoch_loss, epoch_acc = train_gph(gph, optimizer, dataloader, phase='train')
    vld_loss, vld_acc = train_gph(gph, optimizer, dataloader, phase='valid')
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    valid_loss.append(vld_loss)
    valid_acc.append(vld_acc)

