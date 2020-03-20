import torch


def padding(inp, max_seq_len):
    """
    Padding if possible
    :param inp: list<list>, actual words
    :return inp: list<list>, batch_size * max_seq_len 
    :return loc: list, batch_size, location that paddings starting add on
    """
    loc = []
    for i, _ in enumerate(inp):
        inp[i] = inp[i][:max_seq_len]
        if len(inp[i]) < max_seq_len:
            loc.append(len(inp[i]))  # record the location that paddings starting adding on
            padding_size = max_seq_len - len(inp[i])
            inp[i] = inp[i] + ['<PADDING>'] * padding_size
        else:
            loc.append(-1)  # if no need for paddings, record the location as "-1"
    return inp, loc


def embedding(inp, wv_dict):
    """
    :param inp: list<list>, actual word with paddings
    """
    embedding_model = wv_dict.embedding_model
    embedding_dim_ = embedding_model.wv.vector_size
    batch_size = len(inp)
    
    ans = torch.zeros(1, embedding_dim_).float()
    for seq in inp:
        for word in seq:
            embedded_vec = torch.from_numpy(
                wv_dict.index2vector(wv_dict.word2index(word))).float().reshape(1, -1)
            ans = torch.cat((ans, embedded_vec), dim=0)
    ans = ans[1:].view(batch_size, -1, embedding_dim_)
    return ans