import torch
import torch.nn as nn


def init_hidden(batch_size, hidden_dim, device):
    """
    https://github.com/williamSYSU/TextGAN-PyTorch/blob/master/models/generator.py
    """
    h = torch.zeros(3*2, batch_size, hidden_dim)  # layer_num * dir_num
    c = torch.zeros(3*2, batch_size, hidden_dim)  # layer_num * dir_num

    return h.to(device), c.to(device)


class LanguageModel(nn.Module):
    """
    https://github.com/williamSYSU/TextGAN-PyTorch/blob/master/models/generator.py
    """
    def __init__(self, wv_dict, hidden_dim):
        super(LanguageModel, self).__init__()
        
        self.wv_dict = wv_dict
        self.embedding_model_ = self.wv_dict.emb_model
        self.embedding_dim_ = self.embedding_model_.wv.vector_size
        self.vocab_size_ = len(self.embedding_model_.wv.vocab)
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            self.embedding_dim_, 
            self.hidden_dim, 
            num_layers=3, 
            bias=True,
            batch_first=True,
            dropout=0.2,
            bidirectional=True)

        # Input of fc: num_directions * hidden_dim
        self.fc = nn.Linear(2 * self.hidden_dim, self.vocab_size_ + 1, bias=True)  # include paddings
        self.log_softmax = nn.LogSoftmax(dim=-1)
                    
    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * (max_seq_len - 1) * embedding_dim
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        out, (h, c) = self.lstm(inp, hidden)  # out: batch_size * (seq_len - 1) * (dir_num * hidden_dim)
                                              # h, c: (layer_num * dir_num) * batch_size * hidden_dim
        out = self.fc(out)  # batch_size * (seq_len - 1) * (vocab_size + 1), because of paddings
        pred = self.log_softmax(out)
        
        if need_hidden:
            return pred, h
        else:
            return pred