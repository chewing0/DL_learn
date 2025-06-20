'''
该文件为RNN基础的文档处理类的定义，用于在后续代码中复用

核心函数有：
textdeal：文本处理类，读取文本并清洗
seqdataloader：数据加载器，封装了随机和顺序采样
load_data: 加载数据集，返回数据加载器和词汇表对象。
'''

import re
import collections
import torch
import random

from torch import nn
from torch.nn import functional as F

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + 
                                             [token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        return self.token_to_idx['<unk>']

# 实现文本的处理
# 输入：文本的路径
# 输出：文本的分词结果和词汇表对象
class TextDeal:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self._read_and_clean()
        
    def _read_and_clean(self):
        """读取文本文件并清洗成小写英文字符，仅保留字母和空格。"""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).lower() for line in lines]

    def tokenize(self, token_type='char'):
        """按字符或单词级别进行分词。"""
        if token_type == 'char':
            return [list(line) for line in self.text]
        elif token_type == 'word':
            return [line.split() for line in self.text]
        else:
            raise ValueError(f'Unknown token type: {token_type}')

    @staticmethod
    def count_corpus(tokens):
        """统计 token 出现频率，支持 1D 和 2D 列表。"""
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    def load_corpus(self, token_type='word', max_tokens=-1):
        """返回 corpus 索引序列和构建的 Vocab 对象。"""
        tokens = self.tokenize(token_type)
        vocab = Vocab(tokens)
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab


# 随机采样函数
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 顺序采样函数
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# # 封装数据加载器
# class SeqDataLoader:
#     def __init__(self, corpus, batch_size, num_steps, use_random_iter=False):
#         self.corpus = corpus
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.use_random_iter = use_random_iter

#     def __iter__(self):
#         if self.use_random_iter:
#             return seq_data_iter_random(self.corpus, self.batch_size, self.num_steps)
#         else:
#             return seq_data_iter_sequential(self.corpus, self.batch_size, self.num_steps)

def load_corpus(file_path, max_tokens=-1):
    """加载《时光机器》文本数据集，返回分词后的索引序列和词汇表对象。"""
    # file_path = r'learning\data\timemachine.txt'
    # file_path = r'learning\data\shakespeare.txt'
    processor = TextDeal(file_path)
    corpus, vocab = processor.load_corpus(token_type='char', max_tokens=max_tokens)
    return corpus, vocab

class SeqDataLoader:
    '''
    封装加载序列数据的迭代器
    返回：数据加载器和vocab
    '''
    def __init__(self, batch_size, num_steps, use_random_iter, file_path, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.corpus, self.vocab = load_corpus(file_path, max_tokens)
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
def load_data(batch_size, num_steps, use_random_iter=False, file_path=r'learning\data\timemachine.txt', max_tokens=10000):
    """
    加载《时光机器》数据集，返回数据加载器和词汇表对象。
    :param batch_size: 批量大小
    :param num_steps: 每个小批量的时间步数
    :param use_random_iter: 是否使用随机采样
    :param max_tokens: 最大令牌数
    :return: 数据加载器和词汇表对象
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, file_path, max_tokens)
    return data_iter, data_iter.vocab


# # 测试代码
# if __name__ == '__main__':
#     # file_path = r'DeepLearning\data\timemachine.txt'
#     file_path = r'learning\data\timemachine.txt'
#     processor = TextDeal(file_path)
#     corpus, vocab = processor.load_corpus(token_type='char', max_tokens=10000)
#     data_iter = SeqDataLoader(corpus, batch_size=32, num_steps=35, use_random_iter=True)

#     for X, Y in data_iter:
#         print(X.shape, Y.shape)
#         print(X[0], Y[0])
#         break