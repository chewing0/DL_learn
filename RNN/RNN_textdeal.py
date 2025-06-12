# 该文件主要实现动手学深度学习中的文本预处理部分的内容
import re
import random
import torch
import collections

def txt_read(file_path):
    txt = open(file_path).readlines()
    txt = [re.sub('[^A-Za-z]+', ' ', line).lower() for line in txt]
    return txt

def tokenize(text, token_type='char'):
    if token_type == 'char':
        text = [list(line) for line in text]
    elif token_type == 'word':
        text = [line.split() for line in text]
    else:
        print('ERROR: unknown token type: ' + token_type)
    return text

def count_corpus(tokens):
    # 这里的token是1d或者2d列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# vocab
# token_freqs：token及其词频的列表
# idx_to_token：token索引列表（idx->token）
# token_to_idx：token及其索引的字典（token->idx）
class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # 展平tokens列表，如果tokens是二维列表，则将其展平为一维
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # 记录token的词频
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # token及其索引
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

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
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

# 至此完成函数文本的预处理，给出vocab
# 输入：最大token数量 & 分词策略
# 输出：语料库（token索引列表） & 词表（Vocab对象）
# 语料库是一个一维列表，包含了所有的token索引
def load_corpus_time_machine(max_tokens=-1, token_type='word'):
    file_path = r'DeepLearning\data\timemachine.txt'
    text = txt_read(file_path)
    tokens = tokenize(text, token_type)

    vocab = Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine(token_type='word')
print(vocab.token_freqs[:10])
