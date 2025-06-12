# 该文件为RNN基础的文档处理类的定义，用于在后续代码中复用
import re
import collections
import torch
import random

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

# 随机采样函数
def seq_data_iter_random(corpus, batch_size, num_steps, device=None):
    # corpus 是一维的 token 索引列表
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    for i in range(0, len(initial_indices), batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = [data(j) for j in batch_indices]
        Y = [corpus[j + 1: j + 1 + num_steps] for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)

# 顺序采样函数
def seq_data_iter_sequential(corpus, batch_size, num_steps, device=None):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    corpus = corpus[offset: offset + num_tokens]
    corpus = torch.tensor(corpus, device=device)
    corpus = corpus.view(batch_size, -1)

    num_batches = (corpus.shape[1] - 1) // num_steps
    for i in range(num_batches):
        X = corpus[:, i * num_steps: (i + 1) * num_steps]
        Y = corpus[:, i * num_steps + 1: (i + 1) * num_steps + 1]
        yield X, Y

# 封装数据加载器
class SeqDataLoader:
    def __init__(self, corpus, batch_size, num_steps, use_random_iter=False, device=None):
        self.corpus = corpus
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.use_random_iter = use_random_iter
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __iter__(self):
        if self.use_random_iter:
            return seq_data_iter_random(self.corpus, self.batch_size, self.num_steps, self.device)
        else:
            return seq_data_iter_sequential(self.corpus, self.batch_size, self.num_steps, self.device)


# 测试代码
if __name__ == '__main__':
    file_path = r'DeepLearning\data\timemachine.txt'
    processor = TextDeal(file_path)
    corpus, vocab = processor.load_corpus(token_type='word', max_tokens=10000)
    data_iter = SeqDataLoader(corpus, batch_size=32, num_steps=35, use_random_iter=True)

    for X, Y in data_iter:
        print(X.shape, Y.shape)
        print(X[0], Y[0])
        break