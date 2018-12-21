import logging
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
import math
import multi_sieve
import vsm
import norm_neural
import copy

def train(train_data, dev_data, d, meddra_dict, opt, fold_idx, pretrain_model):
    logging.info("train the ensemble normalization model ...")

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

    # rule
    logging.info("init rule-based normer")
    multi_sieve.init(opt, train_data, d)

    # vsm
    logging.info("init vsm-based normer")
    poses = vsm.init_vector_for_dict(word_alphabet, dict_alphabet, meddra_dict)
    # alphabet can share between vsm and neural since they don't change
    # but word_embedding cannot
    vsm_model = vsm.VsmNormer(word_alphabet, copy.deepcopy(word_embedding), embedding_dim, dict_alphabet, poses)
    vsm_train_X = []
    vsm_train_Y = []
    for doc in train_data:
        temp_X, temp_Y = vsm.generate_instances(doc.entities, word_alphabet, dict_alphabet)
        vsm_train_X.extend(temp_X)
        vsm_train_Y.extend(temp_Y)
    vsm_train_loader = DataLoader(vsm.MyDataset(vsm_train_X, vsm_train_Y), opt.batch_size, shuffle=True, collate_fn=vsm.my_collate)
    vsm_optimizer = optim.Adam(vsm_model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    if opt.tune_wordemb == False:
        freeze_net(vsm_model.word_embedding)

    # neural
    logging.info("init neural-based normer")
    neural_model = norm_neural.NeuralNormer(word_alphabet, copy.deepcopy(word_embedding), embedding_dim, dict_alphabet)
    if pretrain_model is not None:
        # neural_model.attn.W.weight.data.copy_(pretrain_model.attn.W.weight.data)
        neural_model.linear.weight.data.copy_(pretrain_model.linear.weight.data)
    neural_train_X = []
    neural_train_Y = []
    for doc in train_data:
        temp_X, temp_Y = norm_neural.generate_instances(doc.entities, word_alphabet, dict_alphabet)
        neural_train_X.extend(temp_X)
        neural_train_Y.extend(temp_Y)
    neural_train_loader = DataLoader(norm_neural.MyDataset(neural_train_X, neural_train_Y), opt.batch_size, shuffle=True, collate_fn=norm_neural.my_collate)
    neural_optimizer = optim.Adam(neural_model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    if opt.tune_wordemb == False:
        freeze_net(neural_model.word_embedding)


    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")
    for idx in range(opt.iter):
        epoch_start = time.time()

        vsm_model.train()
        vsm_train_iter = iter(vsm_train_loader)
        vsm_num_iter = len(vsm_train_loader)

        for i in range(vsm_num_iter):
            x, lengths, y = next(vsm_train_iter)

            l = vsm_model.forward_train(x, lengths, y)

            l.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(vsm_model.parameters(), opt.gradient_clip)
            vsm_optimizer.step()
            vsm_model.zero_grad()


        neural_model.train()
        neural_train_iter = iter(neural_train_loader)
        neural_num_iter = len(neural_train_loader)

        for i in range(neural_num_iter):

            x, lengths, y = next(neural_train_iter)

            y_pred = neural_model.forward(x, lengths)

            l = neural_model.loss(y_pred, y)

            l.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(neural_model.parameters(), opt.gradient_clip)
            neural_optimizer.step()
            neural_model.zero_grad()

        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))


        if opt.dev_file:
            p, r, f = norm_utils.evaluate(dev_data, meddra_dict, vsm_model, neural_model)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            if fold_idx is None:
                torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
                torch.save(neural_model, os.path.join(opt.output, "norm_neural.pkl"))
            else:
                torch.save(vsm_model, os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx+1)))
                torch.save(neural_model, os.path.join(opt.output, "norm_neural_{}.pkl".format(fold_idx + 1)))

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

    if fold_idx is None:
        multi_sieve.finalize(True)
    else:
        if fold_idx == opt.cross_validation-1:
            multi_sieve.finalize(True)
        else:
            multi_sieve.finalize(False)

    if len(opt.dev_file) == 0:
        torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
        torch.save(neural_model, os.path.join(opt.output, "norm_neural.pkl"))

    return best_dev_p, best_dev_r, best_dev_f