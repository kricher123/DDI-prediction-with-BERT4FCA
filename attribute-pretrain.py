
import random
import itertools
import pandas as pd
import numpy as np
import math

from math import sqrt as msqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import torch
import torch.functional as F
from torch import nn
from torch.optim import Adadelta
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import torch_directml
################################# BERT #################################


# the maximum of length of sequences
max_len = 846 * 2 + 3
# the number of tokens (objects or attributes)
max_vocab = 1532 + 418 + 5
# the maximum number of masked tokens
max_pred = 4
# dimension of key, values. the dimension of query and key are the same 
d_k = d_v = 32
# dimension of embedding
d_model = 224  # n_heads * d_k
# dimension of hidden layers
d_ff = d_model * 4

# number of heads
n_heads = 7
# number of encoders
n_layers = 7
# the number of input setences
n_segs = 2

p_dropout = .1

#80% the chosen token is replaced by [mask], 10% is replaced by a random token, 10% do nothing
p_mask = .8
p_replace = .1
p_do_nothing = 1 - p_mask - p_replace

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch_directml.device(torch_directml.default_device())
# device = torch.device("cpu")
 
 

def gelu(x):
    '''
    Two way to implements GELU:
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    or
    0.5 * x * (1. + torch.erf(torch.sqrt(x, 2))) 
    '''
    return .5 * x * (1. + torch.erf(x / msqrt(2.)))

#  create a mask tensor to identify the padding tokens in a batch of sequences
def get_pad_mask(tokens, pad_idx=0):
    '''
    suppose index of [PAD] is zero in word2idx
    the size of input tokens is [batch, seq_len]
    '''
    batch, seq_len = tokens.size()
    pad_mask = tokens.data.eq(pad_idx).unsqueeze(1) #.unsqueeze(1) adds a dimension and turns it to column vectors
    pad_mask = pad_mask.expand(batch, seq_len, seq_len)
    
    # The size of pad_mask is [batch, seq_len, seq_len]
    # The resulting tensor has True where padding tokens are located and False elsewhere.
    
    # print(f'the shape of pad_mask is {pad_mask.shape}')
    return pad_mask
# process input tokens to dense vectors before passing them to encoder.
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        self.seg_emb = nn.Embedding(n_segs, d_model)
        '''
        convert indices into vector embeddings.
        max_vocab can be replaced by formal context object vectors or attribute vectors
        '''
        self.word_emb = nn.Embedding(max_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, seg):
        '''
        x: [batch, seq_len]
        '''
        word_enc = self.word_emb(x)
        
        '''
        maybe positional embedding can be deleted
        '''
        # positional embedding
        # pos = torch.arange(x.shape[1], dtype=torch.long, device=device) # .long: round down
        # pos = pos.unsqueeze(0).expand_as(x) # the shape is [1, seq_len]
        # pos_enc = self.pos_emb(pos)

        seg_enc = self.seg_emb(seg)
        x = self.norm(word_enc + seg_enc)
        return self.dropout(x)
        # return: [batch, seq_len, d_model]
 

 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2) / msqrt(d_k))
        # scores: [batch, n_heads, seq_len, seq_len]
        # fill the positions in the scores tensor where the attn_mask is True with a very large negative value (-1e9). 
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # context: [batch, n_heads, seq_len, d_v]
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        '''
        Q, K, V: [batch, seq_len, d_model]
        attn_mask: [batch, seq_len, seq_len]
        '''
        batch = Q.size(0)
        '''
        split Q, K, V to per head formula: [batch, seq_len, n_heads, d_k]
        Convenient for matrix multiply opearation later
        q, k, v: [batch, n_heads, seq_len, d_k or d_v]
        '''
        per_Q = self.W_Q(Q).view(batch, -1, n_heads, d_k).transpose(1, 2)
        per_K = self.W_K(K).view(batch, -1, n_heads, d_k).transpose(1, 2)
        per_V = self.W_V(V).view(batch, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # context: [batch, n_heads, seq_len, d_v]
        context = ScaledDotProductAttention()(per_Q, per_K, per_V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch, -1, n_heads * d_v)

        # output: [batch, seq_len, d_model]
        output = self.fc(context)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.gelu = gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
# Encoder
# pre-LN is easier to train than post-LN, but if fullly training, post_LN have better result than pre-LN. 

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.enc_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, x, pad_mask):
        '''
        pre-norm
        see more detail in https://openreview.net/pdf?id=B1x8anVFPr

        x: [batch, seq_len, d_model]
        '''
        residual = x
        x = self.norm1(x)
        x = self.enc_attn(x, x, x, pad_mask) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual
# next sentence prediction
# pooled representation of the entire sequence as the [CLS] token representation.
'''
The full connected linear layer improve the result while making the model harder to train.
'''
class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        x: [batch, d_model] (first place output)
        '''
        x = self.fc(x)
        x = self.tanh(x)
        return x
class BERT(nn.Module):
    def __init__(self, n_layers):
        super(BERT, self).__init__()
        self.embedding = Embeddings()
        self.encoders = nn.ModuleList([
            EncoderLayer() for _ in range(n_layers)
        ])

        self.pooler = Pooler()
        
        # next sentence prediction. output is 0 or 1.
        self.next_cls = nn.Linear(d_model, 2)
        self.gelu = gelu
        
        # Sharing weight between some fully connect layer, this will make training easier.
        shared_weight = self.pooler.fc.weight
        self.fc = nn.Linear(d_model, d_model)
        self.fc.weight = shared_weight

        shared_weight = self.embedding.word_emb.weight
        self.word_classifier = nn.Linear(d_model, max_vocab, bias=False)
        self.word_classifier.weight = shared_weight

    def forward(self, tokens, segments, masked_pos):
        output = self.embedding(tokens, segments)
        enc_self_pad_mask = get_pad_mask(tokens)
        for layer in self.encoders:
            output = layer(output, enc_self_pad_mask)
        # output: [batch, max_len, d_model]

        # NSP Task
        '''
        Extracting the [CLS] token representation, 
        passing it through the pooler, 
        and making predictions.
        '''
        hidden_pool = self.pooler(output[:, 0]) # only the [CLS] token
        logits_cls = self.next_cls(hidden_pool)

        # Masked Language Model Task
        '''
        extracting representations of masked positions, 
        passing them through a fully connected layer, 
        applying the GELU activation function, 
        and making predictions using the word classifier
        '''
        # masked_pos: [batch, max_pred] -> [batch, max_pred, d_model]
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, d_model)

        # h_masked: [batch, max_pred, d_model]
        h_masked = torch.gather(output, dim=1, index=masked_pos)
        h_masked = self.gelu(self.fc(h_masked))
        logits_lm = self.word_classifier(h_masked)
        # logits_lm: [batch, max_pred, max_vocab]
        # logits_cls: [batch, 2]

        return logits_cls, logits_lm, hidden_pool
    
################################# DATA PREPARATION #################################

def process_train_intents_from_file(filename) :
    intents = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line based on four blank spaces
            parts = line.split('    ')

            # Extract the right sequence (assuming it's the second part after splitting)
            if len(parts) <= 2:
                intent = parts[0].strip()
                intents.append(intent)
    intents = intents[1:]
    attribute_list = list(set(" ".join(intents).split()))
    sorted_attribute_list = sorted(map(int, attribute_list))
    # print("The number of attributes is",len(sorted_attribute_list))
    
    # Create the object2idx dictionary
    attribute2idx = {'a' + str(attri): int(attri) + 468  for  attri in sorted_attribute_list}
    sorted_attribute_list = list(map(str, sorted_attribute_list ))
    # print(sorted_attribute_list)
    # print(attribute2idx)
    special_tokens = {'[PAD]': max_vocab-4, '[CLS]': max_vocab-3, '[SEP]': max_vocab-2, '[MASK]': max_vocab-1 }

    attribute2idx.update(special_tokens)

    idx2attribute = {idx: attribute for attribute, idx in attribute2idx.items()}
    vocab_size = len(attribute2idx)
    # assert len(attribute2idx) == len(idx2attribute)
    
    modified_intents = [' '.join(['a' + token for token in item.split()]) for item in intents]
    # print(intents)
    # print(modified_intents)
    
    intent_token_list = []
    for intent in modified_intents:
        intent_token_list.append([
            attribute2idx[s] for s in intent.split()
        ])
    
    maxlen = 0
    for intent in intent_token_list :
        maxlen = max(maxlen, len(intent))
    print(maxlen)
    
    return intent_token_list, attribute2idx, modified_intents, sorted_attribute_list

# intent_token_test, attribute2idx, modified_intents = process_test_intents_from_file('icfca-context_concepts.txt') 
# intent_token_train, attribute2idx, modified_intents_train, train_attribute_list = process_train_intents_from_file('BMS-POS-with-missing-part-renumbered_concepts.txt')
intent_token_train, attribute2idx, modified_intents_train, train_attribute_list = process_train_intents_from_file('graphs_subset2/graph_numbered_context.txt')
#print(attribute2idx)
#print(intent_token_train)
#exit()

# padding the token lists to have the same length.
def padding(ids, n_pads, pad_symb=0):
    return ids.extend([pad_symb for _ in range(n_pads)])

def masking_procedure(cand_pos, input_ids, masked_symb='[MASK]'):
    masked_pos = []
    masked_tokens = []
    for pos in cand_pos:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        if random.random() < p_mask:
            input_ids[pos] = masked_symb
        elif random.random() > (p_mask + p_replace):
            rand_word_idx = random.randint(0, max_vocab - 4)
            input_ids[pos] = rand_word_idx

    return masked_pos, masked_tokens
def get_neighbor_samples(extents) :
    n = len(extents)
    samples = []

    dep = np.zeros(shape = (n, n), dtype = np.int32)
    neighbor = np.zeros(shape = (n, n), dtype = np.int32)

    for i in range(n) :
        for j in range(i + 1, n) :
            if set(extents[i]).issubset(set(extents[j])) :
                dep[i][j] = 1
            if set(extents[j]).issubset(set(extents[i])) :
                dep[j][i] = 1

    for i in range(n) :
        se = set([])
        for j in range(n) :
            if j != i :
                if dep[j][i] == 1 :
                    rep = False
                    lst = list(se)
                    for idk, k in enumerate(lst) :
                        if dep[k][j] :
                            se.remove(k)
                            se.add(j)
                            rep = True
                        if dep[j][k] :
                            rep = True
                    if not rep :
                        se.add(j)

        for j in range(n) :
            if j in se :
                samples.append([i, j, True])
            elif random.random() < 0.0018 :
                samples.append([i, j, False])
        
    return samples

all_samples = get_neighbor_samples(intent_token_train)
#print(all_samples)
print(len(all_samples))
#exit()
import pickle

nf = 0
nt = 0
for sample in all_samples :
    extent1, extent2, label = sample
    if label == False :
        nf += 1
    else :
        nt += 1

new_all_samples = []
droprate = nt / nf

for sample in all_samples :
    extent1, extent2, label = sample
    if label == True :
        new_all_samples.append([extent1, extent2, True])
    elif random.random() < droprate :
        new_all_samples.append([extent1, extent2, False])
        
with open('attribute_pretrain_samples.pkl', 'wb') as f:
    pickle.dump(new_all_samples, f)
with open('attribute_pretrain_samples.pkl', 'rb') as f:
    all_samples = pickle.load(f)
print(len(all_samples))

def get_permuted_token_list(tokens, thres = 4) :
    tokens_list = []
    if len(tokens) <= thres :
        permutations = itertools.permutations(tokens)
        tokens_list = [list(p) for p in permutations]
    else :
        for i in range(math.comb(thres, thres)) :
            random.shuffle(tokens)
            tokens_list.append(tokens.copy())
    return tokens_list

# A list of sentences and the desired number of data samples as input.
def make_data(intents, all_samples, word2idx, n_data, num_per_sample = 120):
    batch_data = []
    # positive = negative = 0
    max_len = 0
    len_sentences = len(intents)
    for intent in intents :
        max_len = max(max_len, len(intent))
    max_len = max_len * 2 + 3
    print(max_len)
    for sample in all_samples :
        
        tokens_a_idx = sample[0]
        tokens_b_idx = sample[1]
        tokens_a = intent_token_train[tokens_a_idx]
        tokens_b = intent_token_train[tokens_b_idx]
            
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0 for i in range(
            1 + len(tokens_a) + 1)] + [1 for i in range(1 + len(tokens_b))]

        # Determines the number of positions to mask (n_pred) based on the input sequence length.
        n_pred = min(max_pred, max(1, int(len(input_ids) * .15)))
        cand_pos = [i for i, token in enumerate(input_ids)
                    if token != word2idx['[CLS]'] and token != word2idx['[SEP]']] #exclude special tokens.

        # shuffle all candidate position index, to sampling maksed position from first n_pred
        masked_pos, masked_tokens = masking_procedure(
            cand_pos[:n_pred], input_ids, word2idx['[MASK]'])

        # zero padding for tokens to ensure that the input sequences and segment IDs have the maximum sequence length
        padding(input_ids, max_len - len(input_ids))
        # print("the size of input_ids is " ,len(input_ids))
        padding(segment_ids, max_len - len(segment_ids))
        # print("the size of segment_ids is " ,len(segment_ids))

        # zero padding for mask
        if max_pred > n_pred:
            n_pads = max_pred - n_pred
            padding(masked_pos, n_pads)
            padding(masked_tokens, n_pads)

        # Creating Batch Data:
        batch_data.append(
            [input_ids, segment_ids, masked_tokens, masked_pos, sample[2]])
    
    random.shuffle(batch_data)
    print(len(batch_data))
    return batch_data


class BERTDataset(Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, is_next):
        super(BERTDataset, self).__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.is_next = is_next

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.segment_ids[index], self.masked_tokens[index], self.masked_pos[index], self.is_next[index]
    
################################# PRE-TRAIN BERT #################################
print("HERE")
DO_NSP_TEST = False
batch_size = 22 
lr = 2e-5
epochs = 10
train_samples, test_samples = [], []

if DO_NSP_TEST :
    train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
else :
    train_samples = all_samples

batch_data = make_data(intent_token_train, train_samples, attribute2idx, n_data=len(all_samples))

batch_tensor = [torch.LongTensor(ele) for ele in zip(*batch_data)]
dataset = BERTDataset(*batch_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
model = BERT(n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model.to(device)



print('Entering training process...')
for epoch in range(epochs):
    bat = 0
    loss = 0
    for one_batch in dataloader:
        input_ids, segment_ids, masked_tokens, masked_pos, is_next = [ele.to(device) for ele in one_batch]

        logits_cls, logits_lm, _ = model(input_ids, segment_ids, masked_pos)

        loss_cls = criterion(logits_cls, is_next)
        loss_lm = criterion(logits_lm.view(-1, max_vocab), masked_tokens.view(-1))
        loss_lm = (loss_lm.float()).mean()
        loss = loss_cls + loss_lm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bat += 1
    print(f'Epoch:{epoch} Batch:{bat}\t loss: {loss:.6f}')
    torch.save(model.state_dict(), 'attribute_pretrained.dat')


torch.save(model.state_dict(), 'attribute_pretrained.dat')