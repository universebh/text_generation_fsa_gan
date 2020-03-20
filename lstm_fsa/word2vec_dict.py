import numpy as np


class Word2VecDict:
    def __init__(self, emb_model):
        self.emb_model = emb_model

    def word2onehot(word):
        onehot = onehot = [0] * (len(self.emb_model.wv.vocab) + 1)
        if word == '<PADDING>':
            onehot[0] = 1
            return np.array(onehot, dtype=np.int)
        else:
            try:
                onehot[self.emb_model.wv.vocab[word].index + 1] = 1
            except KeyError:
                onehot[0] = 1
            return np.array(onehot, dtype=np.int)

    def onehot2word(self, onehot):
        index = onehot.index(1)
        if index == 0:
            return '<PADDING>'
        else:
            return self.emb_model.wv.index2word[index - 1]

    def index2word(self, index):
        if index == 0:
            return '<PADDING>'
        else:
            return self.emb_model.wv.index2word[index - 1]

    def word2index(self, word):
        if word == '<PADDING>':
            return 0
        else:
            try:
                return self.emb_model.wv.vocab[word].index + 1
            except KeyError:
                return 0

    def index2vector(self, index):
        if index == 0:
            return np.zeros((self.emb_model.wv.vector_size, ), dtype=np.float)
        else:
            return self.emb_model.wv.__getitem__(index2word(index))

    def vector2index(self, vector):
        # vector: numpy
        if np.count_nonzero(vector) == 0:
            return 0
        else:
            word = self.emb_model.wv.similar_by_vector(vector)[0][0]
            return self.word2index(word)

    def index2onehot(self, index):
        onehot = onehot = [0] * (len(self.emb_model.wv.vocab) + 1)
        onehot[index] = 1
        return np.array(onehot, dtype=np.int)

    def onehot2index(self, onehot):
        return onehot.index(1)

    def vector2onehot(self, vector):
        # vector: numpy
        if np.count_nonzero(vector) == 0:
            word = '<PADDING>'
        else:
            word = self.emb_model.wv.similar_by_vector(vector)[0][0]
        return self.word2onehot(word)

    def onehot2vector(self, onehot):
        word = onehot2word(onehot)
        if word == '<PADDING>':
            return np.zeros((self.emb_model.wv.vector_size, ), dtype=np.float)
        else:
            return self.emb_model.wv.__getitem__(word)