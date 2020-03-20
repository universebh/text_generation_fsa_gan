import torch
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    def __init__(self, data, wv_dict, padding, max_seq_len):
        super(LanguageDataset, self).__init__()
        self.data = data  # list<list>, actual words
        self.wv_dict = wv_dict
        self.embedding_model_ = self.wv_dict.emb_model
        self.padding = padding  # function
        self.max_seq_len = max_seq_len
        
        self.embedding_dim_ = self.embedding_model_.wv.vector_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        selected_data = [self.data[index][::-1]]  # list<one list>, actual words with reversed sequence
        selected_data, padding_loc = self.padding(selected_data, self.max_seq_len)  # list<one list>, actual words with paddings
                                                                                    # list<one list>, padding location
        # max_seq_len - 1
        if padding_loc[0] != -1:
            features = torch.from_numpy(
                self.embedding_model_.wv.__getitem__(selected_data[0][:padding_loc[0]])).float()
            features = torch.cat(
                (features, torch.zeros(self.max_seq_len - padding_loc[0] - 1, self.embedding_dim_).float()), dim=0)
        else:
            features = torch.from_numpy(
                self.embedding_model_.wv.__getitem__(selected_data[0][:-1])).float()
        
        target = torch.tensor([self.wv_dict.word2index(w) for w in selected_data[0][1:]]).float()  
        
        return features, target 
        