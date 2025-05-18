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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device(device)
device = torch_directml.device(torch_directml.default_device())

check = 0
output_objects =[]
count = 0

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

class Embeddings(nn.Module):
    def __init__(self, max_vocab, max_len):
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
    def __init__(self, n_layers, max_vocab, max_len):
        super(BERT, self).__init__()
        self.embedding = Embeddings(max_vocab, max_len)
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
    
#     Extract all extents, modify the form of extents as "o1,o2,..." named as modified_extents
# Change objects to indices in extents, named as extent_token_list. It is a list of INDICES not objects!
# Indices of objects and special tokens are from 1 to 338

def process_train_extents_from_file(filename, max_vocab) :
    extents = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line based on four blank spaces
            parts = line.split('    ')

            # Extract the right sequence (assuming it's the second part after splitting)
            if len(parts) >= 2:
                extent = parts[1].strip()
                extents.append(extent)
    # print("The number of concepts is",len(extents))
    object_list = list(set(" ".join(extents).split()))
    sorted_object_list = sorted(map(int, object_list))
    # print("The number of objects is ",len(sorted_object_list))
    
    # Create the object2idx dictionary
    object2idx = {'o' + str(obj): int(obj)  for  obj in sorted_object_list}
    sorted_object_list = list(map(str, sorted_object_list ))
    # print(sorted_object_list)
    special_tokens = {'[PAD]': max_vocab - 4, '[CLS]': max_vocab - 3, '[SEP]': max_vocab - 2, '[MASK]': max_vocab - 1}

    object2idx.update(special_tokens)
    # print(object2idx) 

    idx2object = {idx: object for object, idx in object2idx.items()}
    vocab_size = len(object2idx)

    assert len(object2idx) == len(idx2object)
    
    modified_extents = [' '.join(['o' + token for token in item.split()]) for item in extents]

    # print(len(modified_extents))
    
    extent_token_list = []
    for extent in modified_extents:
        extent_token_list.append([
            object2idx[s] for s in extent.split(' ')
        ])

    # print(len(extent_token_list))
    return extent_token_list, object2idx, modified_extents, sorted_object_list

#max_vocab_object = 473
max_vocab_object = 1537
# extent_token_test , object2idx , modified_extents , test_object_list  = process_train_extents_from_file('icfca-context_concepts.txt')
extent_token_train, object2idx , modified_extents_train, train_object_list  = process_train_extents_from_file('graphs_subset2/graph_numbered_context.txt', max_vocab_object)
print(len(extent_token_train))

'''
Extract all intents, modify the form of intents as "a1,a2,..." named as modified_intents
Change attributes to indices in intents , named as intent_token_list. It is a list of INDICES not attributes!
Indices of attributes are from 399  (int(attribute)+398)

Notice that index will not be continuous if some attributes are not in the reduced formal context!
'''
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
    attribute2idx = {'a' + str(attri): int(attri) + 162  for  attri in sorted_attribute_list}
    sorted_attribute_list = list(map(str, sorted_attribute_list ))
    # print(sorted_attribute_list)
    # print(attribute2idx)


    idx2attribute = {idx: attribute for attribute, idx in attribute2idx.items()}
    vocab_size = len(attribute2idx)
    assert len(attribute2idx) == len(idx2attribute)

    modified_intents = [' '.join(['a' + token for token in item.split()]) for item in intents]
    # print(intents)
    # print(modified_intents)
    
    intent_token_list = []
    for intent in modified_intents:
        intent_token_list.append([
            attribute2idx[s] for s in intent.split()
        ])
        
    return intent_token_list, attribute2idx, modified_intents, sorted_attribute_list

# intent_token_test, attribute2idx, modified_intents = process_test_intents_from_file('icfca-context_concepts.txt') 
intent_token_train, attribute2idx, modified_intents_train, train_attribute_list = process_train_intents_from_file('graphs_subset2/graph_numbered_context.txt')

concept2idx = object2idx | attribute2idx
#
# print(len(intent_token_list))
# print(intent_token_test)
# print(concept2idx)

extents = []
intents = []
extent_token_new = []
intent_token_new = []

with open('graphs_subset2/graph_numbered_context.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # Split the line based on four blank spaces
        parts = line.split('    ')

        # Extract the right sequence (assuming it's the second part after splitting)
        if len(parts) >= 2:
            extent = parts[1].strip()
            extents.append(extent)
            intent = parts[0].strip()
            intents.append(intent)
            
# print(object2idx.keys())
for extent_str, intent_str in zip(extents, intents) :
    extent_list = extent_str.split(' ')
    intent_list = intent_str.split(' ')
    
    extent_tokens = []
    intent_tokens = []
    
    for obj in extent_list :
        if 'o' + obj in object2idx.keys() :
            extent_tokens.append(object2idx['o' + obj])
        
    for attr in intent_list :
        if 'a' + attr in attribute2idx.keys() :
            intent_tokens.append(attribute2idx['a' + attr])
        
    extent_token_new.append(extent_tokens)
    intent_token_new.append(intent_tokens)

def get_true_permute_pairs(extent_token_list, intent_token_list, tup_len = 3) :
    true_permute_pairs = []
    object_permutes = []
    attribute_permutes = []
    
    distribution = np.zeros(shape = (tup_len + 1, tup_len + 1), dtype = np.float32)
    
    for extent, intent in zip(extent_token_list, intent_token_list) :
        extent_len = len(extent)
        intent_len = len(intent)
        
        for obj_len in range(1, tup_len + 1) :
            if extent_len >= obj_len :
                obj_pmts = [' '.join([str(ele) for ele in list(p)] + ['0' for _ in range(tup_len - obj_len)]) for p in itertools.permutations(extent, obj_len)]
            else :
                obj_pmts = []
                
            for obj_pmt in obj_pmts :
                for attr_len in range(1, tup_len + 1) :
                    if intent_len >= attr_len :
                        attr_pmts = [' '.join([str(ele) for ele in list(p)] + ['0' for _ in range(tup_len - attr_len)]) for p in itertools.permutations(intent, attr_len)]
                    else :
                        attr_pmts
                
                    for attr_pmt in attr_pmts :
                        true_permute_pairs.append(obj_pmt + '-' + attr_pmt)
                        distribution[obj_len][attr_len] += 1

    true_permute_pairs = set(true_permute_pairs)
    
    return true_permute_pairs, distribution

def pad_negative_samples(object2idx, attribute2idx, true_permute_pairs, length_distribution, number) :
    tup_len = length_distribution.shape[0] - 1
    lengths = length_distribution.shape[0] * length_distribution.shape[1]
    length_distribution = length_distribution.reshape(-1)

    print(lengths)
    print(length_distribution)
    selected_permute_pairs = set([])
    
    object_full_list = []
    for obj in object2idx :
        if not '[' in obj :
            object_full_list.append(object2idx[obj])
    attribute_full_list = []
    for attr in attribute2idx :
        attribute_full_list.append(attribute2idx[attr])
    
    negative_samples = []
    while len(negative_samples) < number :
        length_id = np.random.choice(lengths, p=length_distribution)
        obj_length = int(length_id / (tup_len + 1))
        attr_length = length_id % (tup_len + 1)

        obj_list = random.sample(object_full_list, obj_length)
        attr_list = random.sample(attribute_full_list, attr_length)
        
        if obj_length < tup_len :
            obj_list.extend([0 for _ in range(tup_len - obj_length)])
        if attr_length < tup_len :
            attr_list.extend([0 for _ in range(tup_len - attr_length)])
        
        tmp_str = ' '.join([str(x) for x in obj_list]) + '-' + ' '.join([str(y) for y in attr_list])
        
        if tmp_str in true_permute_pairs :
            continue
            
        selected_permute_pairs.add(tmp_str) 
        
        negative_samples.append((obj_list, attr_list, False))
    return negative_samples, selected_permute_pairs

def prepare_list_data(object2idx, attribute2idx, extent_token_list, extent_token_list_new, intent_token_list, intent_token_list_new, tup_len = 3) :
    old_true_permute_pairs, old_distribution = get_true_permute_pairs(extent_token_list, intent_token_list, tup_len)
    new_true_permute_pairs, new_distribution = get_true_permute_pairs(extent_token_list_new, intent_token_list_new, tup_len)
    
    added_true_permute_pairs = new_true_permute_pairs - old_true_permute_pairs
    added_distribution = new_distribution - old_distribution
    
    old_distribution /= np.sum(old_distribution)
    added_distribution /= np.sum(added_distribution)
    
    train_samples = []
    test_samples = []
    
    for perm_str in old_true_permute_pairs :
        perm_substrs = perm_str.split('-')
        obj_list = [int(x) for x in perm_substrs[0].split(' ')]
        attr_list = [int(x) for x in perm_substrs[1].split(' ')]
        train_samples.append((obj_list, attr_list, True))
        
    for perm_str in added_true_permute_pairs :
        perm_substrs = perm_str.split('-')
        obj_list = [int(x) for x in perm_substrs[0].split(' ')]
        attr_list = [int(x) for x in perm_substrs[1].split(' ')]
        test_samples.append((obj_list, attr_list, True))
    
    train_len = len(train_samples)
    test_len = len(test_samples)
    
    train_negative_samples, selected_permute_pairs = pad_negative_samples(object2idx, attribute2idx, old_true_permute_pairs, old_distribution, train_len)
    test_negative_samples, selected_permute_pairs_train = pad_negative_samples(object2idx, attribute2idx, old_true_permute_pairs.union(selected_permute_pairs), added_distribution, test_len)

    train_samples.extend(train_negative_samples)
    test_samples.extend(test_negative_samples)
    
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    return train_samples, test_samples

tup_len = 1

train_labeled_lists, test_labeled_lists = prepare_list_data(object2idx, attribute2idx, extent_token_train, extent_token_new, intent_token_train, intent_token_new, tup_len = tup_len)

# print(len(train_labeled_lists))
# print(len(test_labeled_lists))

# design a MLP for classification task
class MLP(nn.Module):
    def __init__(self, object_pretrained_model, attribute_pretrained_model, embedding_size, hidden_size, output_size, dropout_rate = .1):
        super(MLP, self).__init__()
        
        self.bert_object = object_pretrained_model
        self.bert_attribute = attribute_pretrained_model

        self.fc1 = nn.Linear(embedding_size*2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_object, segments_object, masked_poses_object,inputs_attribute, segments_attribute, masked_poses_attribute):
        _, __, x1 = self.bert_object(inputs_object, segments_object, masked_poses_object)
        _, __, x2 = self.bert_attribute(inputs_attribute, segments_attribute, masked_poses_attribute)
        x = self.fc1(torch.cat((x1,x2), dim=1))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        global count
        if check == 1 :
            for y in inputs_object:
                temp = []
                temp.append(y)
                output_objects.append(temp)
            count = count + 1
            for y in inputs_attribute:
                temp = []
                temp.append(y+2000)
                output_objects.append(temp)
            count = count + 1
        return x
    
def prepare_data(samples):
    inputs = []
    labels = []
    for extent, intent, label in samples:
        inputs.append(extent + intent)
        labels.append(label)
    return torch.tensor(inputs), torch.tensor(labels)

# Set parameters
input_size = 2 * d_model
hidden_size = 477
output_size = 1
learning_rate = 1.7e-5
num_epochs = 5
batch_size = 6


# the maximum of length of extents
object_max_len = 91 # longest extents is 8
# the number of tokens objects
object_max_vocab = 1537
object_pretrained_model = BERT(n_layers, object_max_vocab, object_max_len)

# the maximum of length of sequences
attribute_max_len = 1695
# the number of tokens (objects or attributes)
attribute_max_vocab = 1955
attribute_pretrained_model = BERT(n_layers, attribute_max_vocab, attribute_max_len)

attribute_pretrained_model.load_state_dict(torch.load('attribute_pretrained.dat'))
object_pretrained_model.load_state_dict(torch.load('oo_no_pos_pretrained.dat'))

object_pretrained_model.train()
attribute_pretrained_model.train()

#object_pretrained_model.eval()
#attribute_pretrained_model.eval()

object_pretrained_model.to(device)
attribute_pretrained_model.to(device)

# Instantiate the model, loss function, and optimizer
MLP_model = MLP(object_pretrained_model, attribute_pretrained_model, d_model, hidden_size, output_size, dropout_rate=0.1)
criterion = nn.BCELoss()
optimizer = Adam(MLP_model.parameters(), lr=learning_rate)

# Move model to device
MLP_model = MLP_model.to(device)

# Prepare the data
train_inputs, train_labels = prepare_data(train_labeled_lists)
test_inputs, test_labels = prepare_data(test_labeled_lists)

train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

# Create DataLoader
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

object_pretrained_model.train()
attribute_pretrained_model.train()
MLP_model.train()

for epoch in range(num_epochs):
    # Create tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)

    for inputs, labels in pbar:
        optimizer.zero_grad()
        
        extents, intents = torch.tensor_split(inputs, [tup_len], dim=1)
        
        object_segments = torch.tensor([[0 for _ in i] for i in extents])
        object_masked_poses = torch.tensor([[0 for _ in range(max_pred)] for i in extents])
        
        attribute_segments = torch.tensor([[0 for _ in i] for i in intents])
        attribute_masked_poses = torch.tensor([[0 for _ in range(max_pred)] for i in intents])
        
        extents, intents, labels = extents.to(device), intents.to(device), labels.to(device)
        object_segments, object_masked_poses = object_segments.to(device), object_masked_poses.to(device)
        attribute_segments, attribute_masked_poses = attribute_segments.to(device), attribute_masked_poses.to(device)
        
        outputs = MLP_model(extents, object_segments, object_masked_poses, intents, attribute_segments, attribute_masked_poses)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        # Update tqdm with the current loss
        pbar.set_postfix(loss=loss.item())

MLP_model.eval()
train_extents, train_intents = torch.tensor_split(train_inputs, [tup_len], dim=1)

object_segments = torch.tensor([[0 for _ in i] for i in train_extents])
object_masked_poses = torch.tensor([[0 for _ in range(max_pred)] for i in train_extents])

attribute_segments = torch.tensor([[0 for _ in i] for i in train_intents])
attribute_masked_poses = torch.tensor([[0 for _ in range(max_pred)] for i in train_intents])

object_segments = object_segments.to(device)
attribute_segments = attribute_segments.to(device)

object_masked_poses = object_masked_poses.to(device)
attribute_masked_poses = attribute_masked_poses.to(device)

with torch.no_grad():
    check = 1
    train_outputs = MLP_model(train_extents, object_segments, object_masked_poses, train_intents, attribute_segments, attribute_masked_poses)
    predictions = (train_outputs > 0.5).float().cpu().numpy()
    train_labels_numpy = train_labels.cpu().numpy()

# Convert predictions to binary (0 or 1)
print(torch.is_tensor(train_outputs))
print(type(train_outputs))
torch.save(train_outputs , 'tensor_result.pt')
print(output_objects)#FIX
file_output_objects = open("output_objects.txt" , "w+" , encoding="utf8")

for x in output_objects:
    for y in x:
        tmp = y[0].item()
        file_output_objects.write(str(tmp))
        file_output_objects.write(" ")
    file_output_objects.write("\n")
predictions_binary = (predictions > 0.5).astype(int)

# Compute metrics
accuracy = accuracy_score(train_labels_numpy, predictions_binary)
precision = precision_score(train_labels_numpy, predictions_binary)
recall = recall_score(train_labels_numpy, predictions_binary)
f1 = f1_score(train_labels_numpy, predictions_binary)
auc = roc_auc_score(train_labels_numpy, train_outputs.cpu().numpy())

# Print the results
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')
print(f'AUC: {auc:.3f}')