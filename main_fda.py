import random
import numpy as np
import torch
import os
import logging

from options import opt
import data
import train
import test
from my_utils import makedir_and_clear


logger = logging.getLogger()
if opt.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.info(opt)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

d = data.Data(opt)

if opt.whattodo == 1:
    logging.info("load data ...")
    documents = data.load_data_fda(opt.train_file, True)

    # we use 5-fold cross-validation to evaluate the model
    fold_num = 5
    total_doc_num = len(documents)
    dev_doc_num = total_doc_num // fold_num

    for fold_idx in range(fold_num):
        # debug feili
        if fold_idx ==0 or fold_idx == 1 or fold_idx == 2 or fold_idx == 4:
            continue

        fold_start = fold_idx*dev_doc_num
        fold_end = fold_idx*dev_doc_num+dev_doc_num
        if fold_end > total_doc_num:
            fold_end = total_doc_num

        # debug feili
        # d.train_data = []
        # d.train_data.extend(documents[:fold_start])
        # d.train_data.extend(documents[fold_end:])
        d.train_data = documents[fold_start:fold_end]
        d.dev_data = documents[fold_start:fold_end]

        logging.info("begin fold {}".format(fold_idx))

        logging.info("build alphabet ...")
        d.build_alphabet(d.train_data)
        d.build_alphabet(d.dev_data)
        d.fix_alphabet()

        logging.info("generate instance ...")
        d.train_texts, d.train_Ids = data.read_instance(d.train_data, d.word_alphabet, d.char_alphabet,
                                                        d.label_alphabet)
        d.dev_texts, d.dev_Ids = data.read_instance(d.dev_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)

        logging.info("load pretrained word embedding ...")
        d.pretrain_word_embedding, d.word_emb_dim = data.build_pretrain_embedding(opt.word_emb_file, d.word_alphabet,
                                                                                  opt.word_emb_dim, False)

        train.train(d, opt)

