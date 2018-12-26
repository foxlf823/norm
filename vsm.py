import torch.nn as nn
import logging
from alphabet import Alphabet
from my_utils import random_embedding, freeze_net
import torch
from data import build_pretrain_embedding, my_tokenize, load_data_fda
import numpy as np
import torch.nn.functional as functional
import os
from data_structure import Entity
import norm_utils
from options import opt
import random
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time
from tqdm import tqdm

class VsmNormer(nn.Module):

    def __init__(self, word_alphabet, word_embedding, embedding_dim, dict_alphabet, poses):
        super(VsmNormer, self).__init__()
        self.word_alphabet = word_alphabet
        self.embedding_dim = embedding_dim
        self.word_embedding = word_embedding
        self.dict_alphabet = dict_alphabet
        self.gpu = opt.gpu
        self.poses = poses
        self.dict_size = norm_utils.get_dict_size(dict_alphabet)

        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear.weight.data.copy_(torch.eye(self.embedding_dim))

        if torch.cuda.is_available():
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            self.linear = self.linear.cuda(self.gpu)


    def forward_train(self, mention, lengths, y):

        length = mention.size(1)
        mention_word_emb = self.word_embedding(mention)
        mention_word_emb = mention_word_emb.unsqueeze_(1)
        mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1))
        mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1)

        length = self.poses.size(1)
        pos_word_emb = self.word_embedding(self.poses)
        pos_word_emb = pos_word_emb.unsqueeze_(1)
        pos_word_pool = functional.avg_pool2d(pos_word_emb, (length, 1))
        pos_word_pool = pos_word_pool.squeeze_(1).squeeze_(1)

        m_W = self.linear(mention_word_pool)

        pos_similarity = torch.matmul(m_W, torch.t(pos_word_pool))


        y = y.unsqueeze(-1).expand(-1, self.dict_size)


        a = torch.gather(pos_similarity, 1, y)


        loss = (1-a+pos_similarity).clamp(min=0)

        loss = torch.sum(loss)-1


        return loss

    def forward_eval(self, mention, lengths):
        length = mention.size(1)
        mention_word_emb = self.word_embedding(mention)
        mention_word_emb = mention_word_emb.unsqueeze_(1)
        mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1))
        mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1)

        length = self.poses.size(1)
        pos_word_emb = self.word_embedding(self.poses)
        pos_word_emb = pos_word_emb.unsqueeze_(1)
        pos_word_pool = functional.avg_pool2d(pos_word_emb, (length, 1))
        pos_word_pool = pos_word_pool.squeeze_(1).squeeze_(1)

        m_W = self.linear(mention_word_pool)

        pos_similarity = torch.matmul(m_W, torch.t(pos_word_pool))

        return pos_similarity

    def normalize(self, similarities):

        return functional.softmax(similarities, dim=1)

    def process_one_doc(self, doc, entities, dict):

        Xs, Ys = generate_instances(entities, self.word_alphabet, self.dict_alphabet)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            mention, lengths, _ = next(data_iter)

            similarities = self.forward_eval(mention, lengths)

            similarities = self.normalize(similarities)

            values, indices = torch.max(similarities, 1)

            actual_batch_size = mention.size(0)

            for batch_idx in range(actual_batch_size):
                entity = entities[entity_start+batch_idx]
                norm_id = norm_utils.get_dict_name(self.dict_alphabet, indices[batch_idx].item())
                name = dict[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(name)
                if opt.ensemble == 'sum':
                    entity.norm_confidences.append(similarities[batch_idx].detach().cpu().numpy())
                else:
                    entity.norm_confidences.append(values[batch_idx].item())
                entity.vsm_id = norm_id

            entity_start += actual_batch_size




class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def generate_instances(entities, word_alphabet, dict_alphabet):
    Xs = []
    Ys = []
    dict_size = norm_utils.get_dict_size(dict_alphabet)

    for entity in entities:
        if len(entity.norm_ids) > 0:
            Y = norm_utils.get_dict_index(dict_alphabet, entity.norm_ids[0])
            if Y >= 0 and Y < dict_size:
                pass
            else:
                continue
        else:
            Y = 0

        # mention
        tokens = my_tokenize(entity.name)
        mention = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            mention.append(word_id)

        Xs.append(mention)
        Ys.append(Y)


    return Xs, Ys


# def generate_instances_for_eval(entity, word_alphabet):
#
#     # mention
#     tokens = my_tokenize(entity.name)
#     mention = []
#     for token in tokens:
#         token = norm_utils.word_preprocess(token)
#         word_id = word_alphabet.get_index(token)
#         mention.append(word_id)
#     max_len = len(mention)
#     mentions = [mention]
#     mentions = pad_sequence(mentions, max_len)
#
#     if torch.cuda.is_available():
#         mentions = mentions.cuda(opt.gpu)
#
#     return mentions

def init_vector_for_dict(word_alphabet, dict_alphabet, meddra_dict):

    # pos
    poses = []
    dict_size = norm_utils.get_dict_size(dict_alphabet)
    max_len = 0
    for i in range(dict_size):

        # pos
        concept_name = meddra_dict[norm_utils.get_dict_name(dict_alphabet, i)]
        tokens = my_tokenize(concept_name)
        pos = []
        for token in tokens:
            token = norm_utils.word_preprocess(token)
            word_id = word_alphabet.get_index(token)
            pos.append(word_id)

        if len(pos) > max_len:
            max_len = len(pos)

        poses.append(pos)

    poses = pad_sequence(poses, max_len)

    if torch.cuda.is_available():
        poses = poses.cuda(opt.gpu)

    return poses


def my_collate(batch):
    x, y = zip(*batch)

    lengths = [len(row) for row in x]
    max_len = max(lengths)
    x = pad_sequence(x, max_len)

    lengths = torch.LongTensor(lengths)
    y = torch.LongTensor(y).view(-1)


    if torch.cuda.is_available():
        x = x.cuda(opt.gpu)
        lengths = lengths.cuda(opt.gpu)
        y = y.cuda(opt.gpu)

    return x, lengths, y

def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x


def train(train_data, dev_data, d, meddra_dict, opt, fold_idx):
    logging.info("train the vsm-based normalization model ...")

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
    norm_utils.build_alphabet_from_dict(word_alphabet, meddra_dict)
    norm_utils.build_alphabet(word_alphabet, train_data)
    if opt.dev_file:
        norm_utils.build_alphabet(word_alphabet, dev_data)
    norm_utils.fix_alphabet(word_alphabet)

    if d.config.get('norm_emb') is not None:
        logging.info("load pretrained word embedding ...")
        pretrain_word_embedding, word_emb_dim = build_pretrain_embedding(d.config.get('norm_emb'),
                                                                         word_alphabet,
                                                                              opt.word_emb_dim, False)
        word_embedding = nn.Embedding(word_alphabet.size(), word_emb_dim, padding_idx=0)
        word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        embedding_dim = word_emb_dim
    else:
        logging.info("randomly initialize word embedding ...")
        word_embedding = nn.Embedding(word_alphabet.size(), d.word_emb_dim, padding_idx=0)
        word_embedding.weight.data.copy_(
            torch.from_numpy(random_embedding(word_alphabet.size(), d.word_emb_dim)))
        embedding_dim = d.word_emb_dim


    dict_alphabet = Alphabet('dict')
    norm_utils.init_dict_alphabet(dict_alphabet, meddra_dict)
    norm_utils.fix_alphabet(dict_alphabet)

    logging.info("init_vector_for_dict")
    poses = init_vector_for_dict(word_alphabet, dict_alphabet, meddra_dict)


    vsm_model = VsmNormer(word_alphabet, word_embedding, embedding_dim, dict_alphabet, poses)



    logging.info("generate instances for training ...")
    train_X = []
    train_Y = []

    for doc in train_data:
        temp_X, temp_Y = generate_instances(doc.entities, word_alphabet, dict_alphabet)
        train_X.extend(temp_X)
        train_Y.extend(temp_Y)


    train_loader = DataLoader(MyDataset(train_X, train_Y), opt.batch_size, shuffle=True, collate_fn=my_collate)

    optimizer = optim.Adam(vsm_model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    if opt.tune_wordemb == False:
        freeze_net(vsm_model.word_embedding)

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")

    for idx in range(opt.iter):
        epoch_start = time.time()

        vsm_model.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        for i in range(num_iter):

            x, lengths, y = next(train_iter)

            l = vsm_model.forward_train(x, lengths, y)

            l.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(vsm_model.parameters(), opt.gradient_clip)
            optimizer.step()
            vsm_model.zero_grad()

        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        if opt.dev_file:
            p, r, f = norm_utils.evaluate(dev_data, meddra_dict, vsm_model, None, None, d)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            if fold_idx is None:
                torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
            else:
                torch.save(vsm_model, os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx+1)))

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
        torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))

    return best_dev_p, best_dev_r, best_dev_f





