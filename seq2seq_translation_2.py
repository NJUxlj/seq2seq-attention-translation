import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  
from collections import Counter  
import random  

import unicodedata

from typing import Tuple, List, Dict, Union

import string
import re

import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10


SOS_token=0
EOS_token=1



class Lang():
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1






def unicodeToAscii(s):
    return "".join(
       c for c in unicodedata.normalize('NFD', s)
       if unicodedata.category(c) != 'Mn'
    )





def normalize_string(s):
    s = unicodeToAscii(s.lower().strip())
    s =re.sub(r"([.!?])", r" \1", s)
    s =re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s





def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open("data/%s-%s.txt"%(lang1, lang2), encoding='utf-8' ).read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')[:2]] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)

    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    return input_lang, output_lang, pairs






MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)




def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)




def filterPairs(pairs):
    return [p for p in pairs if filterPair(p)]




def prepareData(lang1:Lang, lang2:Lang, reverse=False):
    '''
        输入：
            lang1: 源语言
            lang2: 目标语言
            reverse: 是否翻转输入输出
        输出：
            input_lang: 源语言的Lang类实例 
            output_lang: 目标语言的Lang类实例 
            pairs: 句子对 :: [(src, tgt), ...]
    '''
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs)) # 从 pairs 列表中随机选择一个句子对




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding= nn.Embedding(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers=5

        self.gru = nn.GRU(
            hidden_size, 
            hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
        )

    def forward(self, input, hidden:torch.Tensor):
        '''
        input.shape = (1, 1)
        '''

        hidden = hidden.expand(self.num_layers, 1, hidden.shape[-1]).contiguous()  
        embedded = self.embedding(input).view(1,1,-1) # shape = (1,1,hidden_size) = (batch_size, seq_len, hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)









class DecoderRNN(nn.Module):
    def __init__(self,output_size, hidden_size):
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.relu = F.relu()
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers = 5,
            batch_first = True,
        )
        self.classifer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)


    def forward(self, x, hidden):
        output = self.embedding(x).view(1,1,-1) # shape = (1,1,hidden_size) = (batch_size, seq_len, hidden_size)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.classifer(output[:,0,:])
        output = self.softmax(output)

        return output, hidden



    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    




class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p = 0.1, max_length = MAX_LENGTH):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn2 = nn.Linear(self.hidden_size * 2, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=5, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    

    def forward(self, input, hidden, encoder_outputs):
        '''
        hidden: 上一时刻的隐状态 shape = (1,1,hidden_size)

        return:
            output: shape = (1, output_size)
            hidden: shape = (1, 1, hidden_size)
            attn_weights: shape = (1, max_length)
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 计算attention weight
        # 公式：e_ij = align(s_{i-1}, h_j)
        # 其中align是一个线性层


        attn_weights = []

        # 将 hidden 调整为正确的形状  
        if hidden.size(0) > 1:  # 如果是多层GRU  shape = (5, 1, hidden_size)
            hidden_for_attn = hidden[-1].unsqueeze(0)  # 只使用最后一层的隐藏状态  
        else:  
            hidden_for_attn = hidden  
        
        # 重复 hidden 状态以匹配序列长度  
        # print("self.hidden_size = ", self.hidden_size)
        hidden_expanded = hidden_for_attn.expand(1, 1, self.hidden_size)  

        hidden_expanded
        # 下面计算attention的代码也可以替换为
        # attn_weights = F.softmax(self.attn(torch.cat((hidden[0], encoder_outputs), dim = -1)), dim = -1)

        for j in range(len(encoder_outputs)):
            #  attn.shape = (2*hidden_size, max_length)
            # unit_to_be_aligned.shape = (1, 2*hidden_size)
            unit_to_be_aligned = torch.cat((hidden_expanded[0], encoder_outputs[j].unsqueeze(0)), dim = -1) # shape = (1, 512)
            # print("unit_to_be_aligned.shape = ", unit_to_be_aligned.shape)
            attn_j = self.attn2(unit_to_be_aligned) # shape = (1, 1)
            attn_j = attn_j.squeeze().squeeze()
            # print("attn_j.shape = ",attn_j.shape)
            attn_weights.append(attn_j)

        attn_weights = torch.tensor(attn_weights, dtype=torch.float, device=device).view(1, -1)
        
        attn_weights = F.softmax(attn_weights, dim = -1) # shape = (1, max_length)


        # 对编码器的输出attention加权求和，得到上下文向量
        # attn_weigghts x encoder_outputs = (1，1, max_length) x (1，max_length, hidden_size) = (1, 1, hidden_size)
        # 公式：c_i = \sum_{j=1}^{T_x} a_{ij}h_j
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 要计算 p(yi|y1, . . . , y_{i-1}, x) = g(y_{i-1}, si, ci), 其中 s_i = f(s_{i-1}, y_{i-1}, c_i)

        # 公式：s_i = f(s_{i-1}, y_{i-1}, c_i)
        # 其中f是一个GRU单元

        # 公式：p(yi|y1,..., y_{i-1}, x) = g(y_{i-1}, si, ci)
        # 其中g是一个线性层


        # 将输入信息(y_{i-1})和上下文信息(c_i)结合使解码器能同时利用局部和全局信息
        output = torch.cat((embedded[0], attn_applied[0]), dim = -1) # shape = (1, 2*hidden_size)

        # 过投影层
        output = self.attn_combine(output).unsqueeze(0) # shape = (1, 1, hidden_size)

        # 过激活函数
        output = F.relu(output)

        output, hidden = self.gru(output, hidden) # s_i = f(s_{i-1}, y_{i-1}, c_i)  shape = (1, 1, hidden_size)

        output = F.log_softmax(self.out(output[0]), dim=-1) # 先过词表投影层 （LM-Head), 再过log-softmax

        return output, hidden, attn_weights
    

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)





def indexesFromSentence(lang:Lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang:Lang, sentence)->torch.LongTensor:
    '''
    sentence: str
    return: tensor, shape = (seq_len, 1)
    '''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair)->Tuple[torch.LongTensor, torch.LongTensor]:
    input_tensor =tensorFromSentence(input_lang, pair[0]) 
    output_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, output_tensor)




teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder: EncoderRNN, decoder:AttnDecoderRNN, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    input_tensor.shape = (seq_len, 1)
    '''
    encoder_hidden = encoder.initHidden() # shape = (1,1,hidden_size) = (batch_size, seq_len, hidden_size)


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device, dtype = torch.float)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder.forward(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0] # shape = (hidden_size, )

    decoder_input = torch.tensor([[SOS_token]], device = device, dtype=torch.long)

    # encoder最后一个隐单元作为decoder的初始隐单元
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attentions = decoder.forward(
                    decoder_input, 
                    decoder_hidden, 
                    encoder_outputs
                )
            loss+= criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else: 
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attentions = decoder.forward(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs
                )
            topv, topi = decoder_output.topk(1)
            token_id = topi.squeeze().detach()
            decoder_input = token_id
            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()


    return loss.item() / target_length




import time, math


def asMinites(s):
    m = math.floor(s/16)
    sec = s - m*60

    return '%dm %ds' % (m, sec)




def timeSince(since, percent):
    now = time.time()
    start = now - since
    end  = start + start * percent
    diff = end - start
    return "%s (- %s)" % (asMinites(start), asMinites(diff))




def trainIters(encoder:EncoderRNN, decoder:AttnDecoderRNN, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_total_loss = 0
    plot_total_loss = 0



    encoder_optimzer = optim.SGD(params=encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(params = decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, 
                     encoder, decoder,
                     encoder_optimzer, decoder_optimizer, 
                     criterion)

        print_total_loss += loss
        plot_total_loss +=loss

        if iter%print_every==0:
            print_loss_avg = print_total_loss / print_every
            print('%s (%d %d) %.4f' % (timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
            print_total_loss = 0


        if iter % plot_every ==0:
            plot_loss_avg = plot_total_loss / plot_every
            plot_total_loss = 0
            pass






def evaluate(encoder:EncoderRNN, decoder:AttnDecoderRNN, sentence, max_length=MAX_LENGTH):
    '''
    这段代码本质上在对一段输入的句子进行推理
    '''
    with torch.no_grad():
        input_tensor =  tensorFromSentence(input_lang,sentence)
        input_length= input_tensor.size(0)
        encoder_hidden = encoder.initHidden()


        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device, dtype=torch.float)


        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder.forward(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0,0]
        
        decoder_input = torch.tensor([[SOS_token]], device = device, dtype=torch.long)

        decoder_hidden = encoder_hidden

        decoder_attentions = torch.zeros(max_length, max_length, device=device, dtype= torch.float)

        decoded_words = []

        for di in range(max_length): # 这里我们方便起见，就指定输出长度为max_length
            decoder_output, decoder_hidden, decoder_attention= decoder.forward()
            decoder_attentions[di] = decoder_attention.data

            topv, topi= decoder_output.data.topk(1)
            if topi.squeeze().detach().item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.squeeze().detach().item()])

            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[:di+1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = " ".join(output_words)
        print('<', output_sentence)
        print('')




if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(output_lang.n_words, hidden_size, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 1000, print_every=100)
    evaluateRandomly(encoder1, attn_decoder1)
