
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from charcnn import CharCNN

class WordRep(nn.Module):
    def __init__(self, data, opt):
        super(WordRep, self).__init__()

        self.gpu = opt.gpu
        self.batch_size = opt.batch_size

        self.char_hidden_dim = opt.char_hidden_dim
        self.char_embedding_dim = opt.char_emb_dim
        self.char_feature = CharCNN(data.char_alphabet.size(), None, self.char_embedding_dim, self.char_hidden_dim, opt.dropout, self.gpu)
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(opt.dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()



    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]

        ## calculate char lstm last hidden
        char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
        char_features = char_features[char_seq_recover]
        char_features = char_features.view(batch_size,sent_len,-1)
        ## concat word and char together
        word_list.append(char_features)
        word_embs = torch.cat([word_embs, char_features], 2)

        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent
        