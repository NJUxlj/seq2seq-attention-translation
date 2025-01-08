
# coding: utf-8

#
# NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
# *******************************************************************************
# **Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
# 
# This is the third and final tutorial on doing "NLP From Scratch", where we
# write our own classes and functions to preprocess the data to do our NLP
# modeling tasks. We hope after you complete this tutorial that you'll proceed to
# learn how `torchtext` can handle much of this preprocessing for you in the
# three tutorials immediately following this one.
# 
# In this project we will be teaching a neural network to translate from
# French to English.
# 
# ::
# 
#     [KEY: > input, = target, < output]
# 
#     > il est en train de peindre un tableau .
#     = he is painting a picture .
#     < he is painting a picture .
# 
#     > pourquoi ne pas essayer ce vin delicieux ?
#     = why not try that delicious wine ?
#     < why not try that delicious wine ?
# 
#     > elle n est pas poete mais romanciere .
#     = she is not a poet but a novelist .
#     < she not not a poet but a novelist .
# 
#     > vous etes trop maigre .
#     = you re too skinny .
#     < you re all alone .
# 
# ... to varying degrees of success.
# 
# This is made possible by the simple but powerful idea of the `sequence
# to sequence network <https://arxiv.org/abs/1409.3215>`__, in which two
# recurrent neural networks work together to transform one sequence to
# another. An encoder network condenses an input sequence into a vector,
# and a decoder network unfolds that vector into a new sequence.
# 
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
# 
# To improve upon this model we'll use an `attention
# mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder
# learn to focus over a specific range of the input sequence.
# 
# **Recommended Reading:**
# 
# I assume you have at least installed PyTorch, know Python, and
# understand Tensors:
# 
# -  https://pytorch.org/ For installation instructions
# -  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
# -  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
# -  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user
# 
# 
# It would also be useful to know about Sequence to Sequence networks and
# how they work:
# 
# -  `Learning Phrase Representations using RNN Encoder-Decoder for
#    Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`__
# -  `Sequence to Sequence Learning with Neural
#    Networks <https://arxiv.org/abs/1409.3215>`__
# -  `Neural Machine Translation by Jointly Learning to Align and
#    Translate <https://arxiv.org/abs/1409.0473>`__
# -  `A Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__
# 
# You will also find the previous tutorials on
# :doc:`/intermediate/char_rnn_classification_tutorial`
# and :doc:`/intermediate/char_rnn_generation_tutorial`
# helpful as those concepts are very similar to the Encoder and Decoder
# models, respectively.
# 
# **Requirements**
# 


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from typing import List, Tuple, Set, Dict, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        '''
        该类是一个某类型语言的自定义词表类
        该类有以下属性：
            name: 语言名称
            word2index: 单词到索引的映射
            word2count: 单词到计数的映射
            index2word: 索引到单词的映射
            n_words: 单词数量

        该类有以下方法：
            addSentence(sentence): 将句子中的所有词添加到词表中
            addWord(word): 将单词添加到词表中
        '''
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):

    '''
    其功能是将 Unicode 字符串转换为纯 ASCII 字符串。

    具体来说，它通过遍历输入字符串 s 中的每个字符 c，
    并使用 unicodedata.normalize('NFD', s) 将其规范化为 NFD 形式（Normalization Form Canonical Decomposition），
    
    然后检查每个字符的 Unicode 类别是否为 'Mn'（Nonspacing Mark）即非间距标记。
    如果字符不是 'Mn' 类别，则将其保留并加入到结果字符串中。最后，函数返回转换后的纯 ASCII 字符串。

    这个函数的作用是去除字符串中的重音符号和其他非 ASCII 字符，
    使得字符串更适合用于自然语言处理任务，例如语言模型训练或文本分析。


    NFD（Normalization Form Canonical Decomposition）
    
        是 Unicode 规范化的一种形式，
        它将 Unicode 字符串分解为其基本字符和组合字符。
        具体来说，NFD 形式将一个字符分解为其基本字符和一个或多个组合字符，
        这些组合字符用于表示重音符号、变音符号等。

        例如，字符 "é" 可以表示为基本字符 "e" 和组合字符 "´"。
        在 NFD 形式中，这个字符将被分解为 "e" 和 "´"。

        NFD 形式的规范化有助于确保 Unicode 字符串在不同的系统和应用程序中具有一致的表示形式，
        从而避免因字符编码不同而导致的问题。
        在自然语言处理任务中，NFD 形式的规范化也有助于确保字符串的一致性和可处理性。



    在 Unicode 字符分类中，'Mn' 代表 "Nonspacing Mark"，即非间距标记。
        这些字符通常是与其他字符组合使用的变音符号或重音符号，它们不占用额外的空间，
        而是附加在基本字符上以改变其发音或意义。
        
        例如，在法语单词 "naïve" 中，字母 "i" 上的两点就是一个非间距标记。
        在 Unicode 规范化过程中，这些标记通常会被分离出来，以便更方便地处理和比较字符串。
    
    '''

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )




# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    '''
    normalizeString 函数的作用是将输入的字符串转换为小写，
    去除两端的空白字符，
    将标点符号与单词分开，
    并去除非字母和标点符号的字符。
    这样处理后的字符串更适合作为自然语言处理任务的输入。
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s) # r" \1"表示将匹配到的标点符号替换为一个空格加上该标点符号本身。
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
    # 为了快速展示结果，我们减少一下使用的训练数据
    # pairs = pairs[:10000]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).

'''
#因为有很多例句，我们想训练
#很快，我们会将数据集修剪为相对较短并且
#简单的句子。这里的最大长度是10个单词（包括结束标点符号），
# 我们正在过滤翻译成形式“我是”或“他是”等（占撇号替换早些时候）。
'''

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
    return [pair for pair in pairs if filterPair(pair)]


# The full process for preparing the data is:
# 
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs


def prepareData(lang1, lang2, reverse=False):
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


# The Seq2Seq Model
# =================
# 
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
# 
# A `Sequence to Sequence network <https://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
# 
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
# 
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
# 
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
# 
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
# 
# 
# 

# The Encoder
# -----------
# 
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
# 



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # input.shape = 
        embedded = self.embedding(input).view(1, 1, -1) # embeded.shape = (1, 1, hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden) 
        # output.shape = (1, 1, hidden_size)
        # hidden.shape = (1, 1, hidden_size)
        return output, hidden

    def initHidden(self):
        '''
        为什么要这么写呢？在序列到序列（Seq2Seq）模型中，编码器（Encoder）需要一个初始的隐藏状态来开始处理输入序列。
        这个初始的隐藏状态通常被设置为全零，因为在处理序列的开始时，我们没有任何先前的信息可以传递。
        通过将隐藏状态初始化为全零，我们可以确保模型在开始处理序列时处于一个中性的状态。
        '''
        # shape = (1, 1, hidden_size) = (batch_size, seq_len, hidden_size)
        return torch.zeros(1, 1, self.hidden_size, device=device)


# The Decoder
# -----------
# 
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
# 
# 
# 

# Simple Decoder
# ^^^^^^^^^^^^^^
# 
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
# 
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
# 

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        '''
        #在解码的每一步，解码器都会得到一个输入令牌和
        #隐藏状态。初始输入标记是字符串的开始"<SOS>"
        #标记，第一个隐藏状态是上下文向量（编码器的
        #最后隐藏状态）。
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''
        input.shape = (batch_size, seq_len, hidden_size) = (1, 1, hidden_size)
        # 每次处理源语言序列中的一个词  

        Encoder的input是源语言序列（比如中文句子）。例如，如果我们要将"我爱你"翻译成英文，那么Encoder的input就是"我爱你"这个序列。
        '''
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
# 
# 
# 

# Attention Decoder
# ^^^^^^^^^^^^^^^^^
# 
# If only the context vector is passed between the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
# 
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
# 

# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
# 


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        '''
        注意力权重表示解码器在生成当前输出时应该关注输入序列的哪些部分。
        '''
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # output_size == vocab_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # attn实际上就是论文中的Alignment model
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
        input: 就decoder上一步的输出 shape = (1, 1)    
        encoder_output: encoder输出的context向量 shape = (1, max_length, hidden_size)

        在标准的seq2seq with attention模型中，Encoder会输出两个重要的张量：

        encoder_outputs: 所有时间步的隐藏状态集合
        final_hidden: 最后一个时间步的隐藏状态

        # 每次处理目标语言序列中的一个词  

        Decoder的input是目标语言序列（比如英文句子）。继续上面的例子，如果我们要将"我爱你"翻译成"I love you"，那么Decoder的input就是"I love you"这个序列。
        '''
        embedded = self.embedding(input).view(1, 1, -1) # shape = (1,1,hidden_size)
        embedded = self.dropout(embedded)

        # torch.cat((embedded[0], hidden[0]), 1)   shape = (1, hidden_size * 2)
        # 将当前输入和隐状态链接起来
        '''
        这一步将当前时间步的嵌入向量（embedded）和上一个时间步的隐藏状态（hidden）在第二个维度上进行拼接。

        这个操作的目的是为了将当前输入的信息和上一个时间步的隐藏状态信息合并，合并是为了对两个vector的信息进行对齐
        以便后续的注意力机制能够综合考虑这两部分信息来计算注意力权重。

        这对应论文中的公式：

        e_{ij} = a(s_{i-1}, h_j)  
        α_{ij} = softmax(e_{ij})  
        其中：

        s_{i-1} 是上一时刻的解码器隐藏状态
        h_j 是编码器输出序列中的第j个向量
        j 取 0 ~ max_length-1 
        '''
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) # shape = (1, max_length)
        

        # 写法2
        # # 计算注意力分数  
        # attn_scores = []  
        # for j in range(encoder_outputs.size(0)):  # 遍历所有encoder输出  
        #     # 对每个encoder隐藏状态计算注意力分数  
        #     score = self.attn(torch.cat((hidden[0], encoder_outputs[j]), 1))  
        #     attn_scores.append(score)  
        
        # # 将分数转换为概率分布  
        # attn_weights = F.softmax(torch.cat(attn_scores), dim=1) 
        '''
        最终，attn_weights 是一个形状为 (1, max_length) 的张量，
        其中每个元素表示解码器在生成当前输出时应该关注输入序列的对应位置的概率。
        这些权重将在后续步骤中用于计算注意力加权的上下文向量
        （attention-weighted context vector）。
        '''

        # 将注意力权重应用到编码器的输出上，得到注意力加权的上下文向量。
        # bmm: 批量矩阵乘法
        '''
        计算attn_applied, 也就是上下文向量
        这一步使用注意力权重对编码器输出进行加权求和，得到上下文向量，对应论文中的公式：
        c_i = Σ_j α_{ij}h_j  

        attn_applied.shape = (1, 1, max_length) x (1, max_length, hidden_size) = (1, 1, hidden_size)
        '''
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), 
                                 encoder_outputs.unsqueeze(0)) 
        # 将输入嵌入和上下文向量拼接
        '''
        embedded.shape = (1, 1, hidden_size)
        attn_applied.shape = (1, 1, hidden_size)
        '''

        '''
        以下公式的作用：解码器状态更新
        s_i = f(s_{i-1}, y_{i-1}, c_i)  

        为什么需要s_i, 实际上就是为了计算: p(yi|y1, . . . , y_{i-1}, x) = g(y_{i-1}, si, ci), , 其中 s_i = f(s_{i-1}, y_{i-1}, c_i)

        # 公式：s_i = f(s_{i-1}, y_{i-1}, c_i)
        # 其中f是一个GRU单元

        # 公式：p(yi|y1,..., y_{i-1}, x) = g(y_{i-1}, si, ci)
        # 其中g是一个线性层
        '''
        # 将输入信息(y_{i-1})和上下文信息(c_i)结合使解码器能同时利用局部和全局信息
        output = torch.cat((embedded[0], attn_applied[0]), 1) # shape = (1, hidden_size * 2)
        # 过投影层
        output = self.attn_combine(output).unsqueeze(0) # shape = (1, 1, hidden_size)

        output = F.relu(output)
        # hidden.shape = (1, 1, hidden_size)
        output, hidden = self.gru(output, hidden) # s_i = f(s_{i-1}, y_{i-1}, c_i) shape = (1, 1, hidden_size)

        # 使用log_softmax而不是普通softmax，这在训练时能提供更好的数值稳定性。
        output = F.log_softmax(self.out(output[0]), dim=1) # shape = (1, output_size) # 先过词表投影层 （LM-Head), 再过log-softmax
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# Training
# ========
# 
# Preparing Training Data
# -----------------------
# 
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
# 



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# Training the Model
# ------------------
# 
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
# 
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.
# 
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
# 
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
# 



teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder:EncoderRNN, decoder:AttnDecoderRNN, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device) # shape = (max_length, hidden_size)

    loss = 0

    for ei in range(input_length):
        # encoder_output.shape = (1, 1, hidden_size)
        # encoder_hidden.shape = (1, 1, hidden_size)
        encoder_output, encoder_hidden = encoder.forward(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward(
                decoder_input, decoder_hidden, encoder_outputs)
            
            # decoder_output.shape = (1, output_size)
            topv, topi = decoder_output.topk(1) # topk函数用于返回张量中前k个最大的值及其对应的索引。
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """
    since: time.time()
    percent: float

    计算从某个起始时间 `since` 到当前时间的时间间隔，并根据已经完成的百分比 `percent` 估算剩余时间。

    时间戳是从某个特定的起始时间（通常是1970年1月1日00:00:00 UTC）到当前时间的秒数。

    参数: 
    since (float): 起始时间的时间戳。
    percent (float): 已经完成的百分比。

    返回:
    str: 一个字符串，格式为 "已用时间 (- 剩余时间)"，其中时间以分钟为单位。
    """
    now = time.time() # 当前时间的时间戳(结束时间)
    s = now - since # 开始时间
    es = s / (percent) # 预计结束时间
    rs = es - s # 剩余时间
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# The whole training process looks like this:
# 
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
# 
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.


def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs:List[Tuple[torch.LongTensor]] = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)


# Plotting results
# ----------------
# 
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

# Evaluation
# ==========
# 
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
# 
'''
#评估大多与培训相同，但没有目标所以
#我们只需为每个步骤将解码器的预测反馈给自身。
#每次它预测一个单词时，我们都会将其添加到输出字符串中，如果它
#预测我们停在那里的EOS代币。我们还存储解码器的
#注意输出以供稍后显示。
'''

def evaluate(encoder: EncoderRNN, decoder: AttnDecoderRNN, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            # encoder_output.shape = (1, 1, hidden_size)
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS shape = (1, 1)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # decoder_attention.shape = (1, max_length)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data # tensor.data 属性用于获取张量的数据部分，而不包括其梯度信息。
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
# 


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# Training and Evaluating
# =======================
# 
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
# 
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
# 
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
# 

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 1000, print_every=100)

evaluateRandomly(encoder1, attn_decoder1)
