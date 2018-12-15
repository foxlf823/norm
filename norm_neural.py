import logging
import torch.nn as nn
from alphabet import Alphabet
from options import opt
import norm_utils
from data import build_pretrain_embedding, my_tokenize, load_data_fda
from my_utils import random_embedding, freeze_net
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import time
import os
from data_structure import Entity
import torch.nn.functional as functional

class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if torch.cuda.is_available():
            idxes = idxes.cuda(opt.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output

class NeuralNormer(nn.Module):

    def __init__(self, word_alphabet, word_embedding, embedding_dim, dict_alphabet):
        super(NeuralNormer, self).__init__()
        self.word_alphabet = word_alphabet
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.dict_alphabet = dict_alphabet
        self.gpu = opt.gpu

        self.attn = DotAttentionLayer(self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, norm_utils.get_dict_size(self.dict_alphabet), bias=False)
        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            self.attn = self.attn.cuda(self.gpu)
            self.linear = self.linear.cuda(self.gpu)
            self.criterion = self.criterion.cuda(self.gpu)

    def forward(self, x, lengths):
        x = self.word_embedding(x)
        x = self.attn((x, lengths))
        x = self.linear(x)

        return x

    def loss(self, y_pred, y_gold):

        return self.criterion(y_pred, y_gold)



    def normalize(self, y_pred):

        return functional.softmax(y_pred, dim=1)

    def process_one_doc(self, doc, entities, dict):

        Xs, Ys = generate_instances(entities, self.word_alphabet, self.dict_alphabet)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            x, lengths, _ = next(data_iter)

            y_pred = self.forward(x, lengths)

            y_pred = self.normalize(y_pred)

            values, indices = torch.max(y_pred, 1)

            actual_batch_size = lengths.size(0)
            # current = entity_start

            for batch_idx in range(actual_batch_size):
                entity = entities[entity_start+batch_idx]
                norm_id = norm_utils.get_dict_name(self.dict_alphabet, indices[batch_idx].item())
                name = dict[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(name)
                entity.norm_confidences.append(values[batch_idx].item())

            # while current < entity_start+actual_batch_size:
            #     entity = entities[current]
            #     norm_id = norm_utils.get_dict_name(self.dict_alphabet, indices[current].item())
            #     name = dict[norm_id]
            #     entity.norm_ids.append(norm_id)
            #     entity.norm_names.append(name)
            #     entity.norm_confidences.append(values[current].item())
            #     current += 1

            entity_start += actual_batch_size




class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def generate_instances(entities, word_alphabet, dict_alphabet):
    Xs = []
    Ys = []

    for entity in entities:
        if len(entity.norm_ids) > 0:
            Y = norm_utils.get_dict_index(dict_alphabet, entity.norm_ids[0])  # use the first id to generate instance
            if Y >= 0 and Y < norm_utils.get_dict_size(dict_alphabet):  # for tac, can be none or oov ID
                Ys.append(Y)
            else:
                continue
        else:
            Ys.append(0)


        tokens = my_tokenize(entity.name)
        word_ids = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            word_ids.append(word_id)

        Xs.append(word_ids)



    return Xs, Ys

def my_collate(batch):
    x, y = zip(*batch)

    x, lengths, y = pad(x, y)

    if torch.cuda.is_available():
        x = x.cuda(opt.gpu)
        lengths = lengths.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
    return x, lengths, y

def pad(x, y):
    tokens = x

    lengths = [len(row) for row in tokens]
    max_len = max(lengths)

    tokens = pad_sequence(tokens, max_len)
    lengths = torch.LongTensor(lengths)

    y = torch.LongTensor(y).view(-1)


    return tokens, lengths, y


def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x


def train(train_data, dev_data, d, meddra_dict, opt, fold_idx):
    logging.info("train the neural-based normalization model ...")

    external_train_data = []
    if d.config.get('norm_ext_corpus') is not None:
        for k, v in d.config['norm_ext_corpus'].items():
            if k == 'tac':
                external_train_data.extend(load_data_fda(v['path'], True, v.get('types'), v.get('types'), False, True))
            else:
                raise RuntimeError("not support external corpus")
    if len(external_train_data) != 0:
        train_data.extend(external_train_data)

    logging.info("build alphabet ...")
    word_alphabet = Alphabet('word')
    norm_utils.build_alphabet(word_alphabet, train_data)
    if opt.dev_file:
        norm_utils.build_alphabet(word_alphabet, dev_data)

    norm_utils.build_alphabet_from_dict(word_alphabet, meddra_dict)
    norm_utils.fix_alphabet(word_alphabet)


    if d.config.get('norm_emb') is not None:
        logging.info("load pretrained word embedding ...")
        pretrain_word_embedding, word_emb_dim = build_pretrain_embedding(d.config.get('norm_emb'),
                                                                         word_alphabet,
                                                                              opt.word_emb_dim, False)
        word_embedding = nn.Embedding(word_alphabet.size(), word_emb_dim)
        word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        embedding_dim = word_emb_dim
    else:
        logging.info("randomly initialize word embedding ...")
        word_embedding = nn.Embedding(word_alphabet.size(), d.word_emb_dim)
        word_embedding.weight.data.copy_(
            torch.from_numpy(random_embedding(word_alphabet.size(), d.word_emb_dim)))
        embedding_dim = d.word_emb_dim



    dict_alphabet = Alphabet('dict')
    norm_utils.init_dict_alphabet(dict_alphabet, meddra_dict)
    norm_utils.fix_alphabet(dict_alphabet)

    neural_model = NeuralNormer(word_alphabet, word_embedding, embedding_dim, dict_alphabet)

    train_X = []
    train_Y = []
    for doc in train_data:
        temp_X, temp_Y = generate_instances(doc.entities, word_alphabet, dict_alphabet)
        train_X.extend(temp_X)
        train_Y.extend(temp_Y)

    train_loader = DataLoader(MyDataset(train_X, train_Y), opt.batch_size, shuffle=True, collate_fn=my_collate)


    optimizer = optim.Adam(neural_model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    if opt.tune_wordemb == False:
        freeze_net(neural_model.word_embedding)

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")

    for idx in range(opt.iter):
        epoch_start = time.time()

        neural_model.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        for i in range(num_iter):

            x, lengths, y = next(train_iter)

            y_pred = neural_model.forward(x, lengths)

            l = neural_model.loss(y_pred, y)

            l.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(neural_model.parameters(), opt.gradient_clip)
            optimizer.step()
            neural_model.zero_grad()

        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        if opt.dev_file:
            p, r, f = norm_utils.evaluate(dev_data, meddra_dict, neural_model)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            if fold_idx is None:
                torch.save(neural_model, os.path.join(opt.output, "norm_neural.pkl"))
            else:
                torch.save(neural_model, os.path.join(opt.output, "norm_neural_{}.pkl".format(fold_idx+1)))

            best_dev_f = f
            best_dev_p = p
            best_dev_r = r

            bad_counter = 0
        else:
            bad_counter += 1

        if len(opt.dev_file) != 0 and bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info("train finished")

    if len(opt.dev_file) == 0:
        torch.save(neural_model, os.path.join(opt.output, "norm_neural.pkl"))

    return best_dev_p, best_dev_r, best_dev_f



