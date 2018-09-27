
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
    d.train_data = data.loadData(opt.train_file)
    d.dev_data = data.loadData(opt.dev_file)
    if opt.test_file:
        d.test_data = data.loadData(opt.test_file)

    logging.info("build alphabet ...")
    d.build_alphabet(d.train_data)
    d.build_alphabet(d.dev_data)
    if opt.test_file:
        d.build_alphabet(d.test_data)

    d.fix_alphabet()

    logging.info("generate instance ...")
    d.train_texts, d.train_Ids = data.read_instance(d.train_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)
    d.dev_texts, d.dev_Ids = data.read_instance(d.dev_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)
    if opt.test_file:
        d.test_texts, d.test_Ids = data.read_instance(d.test_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)

    logging.info("load pretrained word embedding ...")
    d.pretrain_word_embedding, d.word_emb_dim = data.build_pretrain_embedding(opt.word_emb_file, d.word_alphabet, opt.word_emb_dim, False)

    makedir_and_clear(opt.output)

    train.train(d, opt)

    d.clear() # clear some data due it's useless when test
    d.save(os.path.join(opt.output, "data.pkl"))

else:
    d.load(os.path.join(opt.output, "data.pkl"))

    test.test(d, opt)


