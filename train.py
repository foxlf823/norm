from seqmodel import SeqModel
import torch.optim as optim

import torch
import time
import random
import logging

import os
import my_utils
from my_utils import batchify_with_label, evaluate




def train(data, opt):
    model = SeqModel(data, opt)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    if opt.tune_wordemb == False:
        my_utils.freeze_net(model.word_hidden.wordrep.word_embedding)

    best_dev = -10
    bad_counter = 0

    for idx in range(opt.iter):
        epoch_start = time.time()

        random.shuffle(data.train_Ids)

        model.train()
        model.zero_grad()
        batch_size = opt.batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):

            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, opt.gpu)

            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char,
                                                          batch_charlen, batch_charrecover, batch_label, mask)

            loss.backward()
            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
            optimizer.step()
            model.zero_grad()

        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        _, _, p, r, f, _, _ = evaluate(data, opt, model, "dev", True)
        logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))

        if f > best_dev:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev))

            torch.save(model.state_dict(), os.path.join(opt.output, "model.pkl"))
            best_dev = f

            if opt.test_file:
                _, _, p, r, f, _, _ = evaluate(data, opt, model, "test", True, opt.nbest)
                logging.info("Test: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))

            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info("train finished")





