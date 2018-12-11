import torch.nn as nn
import logging
from alphabet import Alphabet
from options import opt
from my_utils import normalize_word, random_embedding
import torch
from data import build_pretrain_embedding, my_tokenize, load_data_fda
import numpy as np
import torch.nn.functional as functional
import os
from data_structure import Entity


class VsmNormer(nn.Module):

    def __init__(self):
        super(VsmNormer, self).__init__()
        self.word_alphabet = Alphabet('word')
        self.embedding_dim = None
        self.word_embedding = None
        self.dict_alphabet = Alphabet('dict')
        self.dict_embedding = None


    def build_alphabet(self, data):
        for document in data:
            for sentence in document.sentences:
                for token in sentence:
                    word = token['text']
                    self.word_alphabet.add(word_preprocess(word))

    def build_dict_alphabet(self, meddra_dict):
        for concept_id, concept_name in meddra_dict.items():
            tokens = my_tokenize(concept_name)
            for word in tokens:
                self.word_alphabet.add(word_preprocess(word))

    def fix_alphabet(self):
        self.word_alphabet.close()

    def get_dict_index(self, concept_id):
        index = self.dict_alphabet.get_index(concept_id)-2 # since alphabet begin at 2
        return index

    def get_dict_name(self, concept_index):
        name = self.dict_alphabet.get_instance(concept_index+2)
        return name

    def batch_name_to_ids(self, name):
        tokens = my_tokenize(name)
        length = len(tokens)
        tokens_id = np.zeros((1, length), dtype=np.int)
        for i, word in enumerate(tokens):
            word = word_preprocess(word)
            tokens_id[0][i] = self.word_alphabet.get_index(word)

        return torch.from_numpy(tokens_id)


    def init_vector_for_dict(self, meddra_dict):
        self.dict_embedding = nn.Embedding(len(meddra_dict), self.embedding_dim)

        for concept_id, concept_name in meddra_dict.items():
            self.dict_alphabet.add(concept_id)
            with torch.no_grad():
                tokens_id = self.batch_name_to_ids(concept_name)
                length = tokens_id.size(1)
                emb = self.word_embedding(tokens_id)
                emb = emb.unsqueeze_(1)
                pool = functional.avg_pool2d(emb, (length, 1))
                index = self.get_dict_index(concept_id)
                self.dict_embedding.weight.data[index] = pool[0][0]

    def compute_similarity(self, mention_rep, concep_rep):
        # mention_rep is (batch, emb_dim) and concep_rep is (concept_num, emb_dim)
        mention_rep_norm = torch.norm(mention_rep, 2, 1, True)  # batch 1
        concep_rep_norm = torch.norm(concep_rep, 2, 1, True)  # concept 1
        a = torch.matmul(mention_rep_norm, torch.t(concep_rep_norm)) # batch, concept
        a = a.clamp(min=1e-8)

        b = torch.matmul(mention_rep, torch.t(concep_rep)) # batch, concept

        return b / a


    def forward(self, mention_word_ids):
        length = mention_word_ids.size(1)
        mention_word_emb = self.word_embedding(mention_word_ids)
        mention_word_emb = mention_word_emb.unsqueeze_(1)
        mention_word_pool = functional.avg_pool2d(mention_word_emb, (length, 1)) # batch,1,1,100
        mention_word_pool = mention_word_pool.squeeze_(1).squeeze_(1) # batch,100

        # similarities = torch.t(torch.matmul(self.dict_embedding.weight.data, torch.t(mention_word_pool))) # batch, dict
        similarities = self.compute_similarity(mention_word_pool, self.dict_embedding.weight.data)

        values, indices = torch.max(similarities, 1)

        return values, indices


def word_preprocess(word):
    if opt.number_normalized:
        word = normalize_word(word)
    word = word.lower()
    return word

def train(train_data, dev_data, d, meddra_dict, opt, fold_idx):
    logging.info("train the vsm-based normalization model ...")

    external_train_data = []
    if d.config.get('norm_ext_corpus') is not None:
        for k, v in d.config['norm_ext_corpus'].items():
            if k == 'tac':
                external_train_data.extend(load_data_fda(v['path'], True, v.get('types'), v.get('types')))
            else:
                raise RuntimeError("not support external corpus")
    if len(external_train_data) != 0:
        train_data.extend(external_train_data)

    vsm_model = VsmNormer()

    logging.info("build alphabet ...")
    vsm_model.build_alphabet(train_data)
    if opt.dev_file:
        vsm_model.build_alphabet(dev_data)
    vsm_model.build_dict_alphabet(meddra_dict)
    vsm_model.fix_alphabet()

    if d.config.get('norm_vsm_emb') is not None:
        logging.info("load pretrained word embedding ...")
        pretrain_word_embedding, word_emb_dim = build_pretrain_embedding(d.config.get('norm_vsm_emb'),
                                                                              vsm_model.word_alphabet,
                                                                              opt.word_emb_dim, False)
        vsm_model.word_embedding = nn.Embedding(vsm_model.word_alphabet.size(), word_emb_dim)
        vsm_model.word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        vsm_model.embedding_dim = word_emb_dim
    else:
        logging.info("randomly initialize word embedding ...")
        vsm_model.word_embedding = nn.Embedding(vsm_model.word_alphabet.size(), d.word_emb_dim)
        vsm_model.word_embedding.weight.data.copy_(
            torch.from_numpy(random_embedding(vsm_model.word_alphabet.size(), d.word_emb_dim)))
        vsm_model.embedding_dim = d.word_emb_dim

    logging.info("init_vector_for_dict")
    vsm_model.init_vector_for_dict(meddra_dict)

    vsm_model.train()

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    if opt.dev_file:
        p, r, f = evaluate(dev_data, meddra_dict, vsm_model)
        logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
    else:
        f = best_dev_f

    if f > best_dev_f:
        logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

        if fold_idx is None:
            logging.info("save model to {}".format(os.path.join(opt.output, "vsm.pkl")))
            torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))
        else:
            logging.info("save model to {}".format(os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx + 1))))
            torch.save(vsm_model, os.path.join(opt.output, "vsm_{}.pkl".format(fold_idx + 1)))

        best_dev_f = f
        best_dev_p = p
        best_dev_r = r


    logging.info("train finished")

    if len(opt.dev_file) == 0:
        logging.info("save model to {}".format(os.path.join(opt.output, "vsm.pkl")))
        torch.save(vsm_model, os.path.join(opt.output, "vsm.pkl"))

    return best_dev_p, best_dev_r, best_dev_f


def evaluate(documents, meddra_dict, vsm_model):
    vsm_model.eval()

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    for document in documents:

        # copy entities from gold entities
        pred_entities = []
        for gold in document.entities:
            pred = Entity()
            pred.id = gold.id
            pred.type = gold.type
            pred.spans = gold.spans
            pred.section = gold.section
            pred.name = gold.name
            pred_entities.append(pred)

        process_one_doc(document, pred_entities, meddra_dict, vsm_model)

        ct_gold += len(document.entities)
        ct_predicted += len(pred_entities)
        for idx, pred in enumerate(pred_entities):
            gold = document.entities[idx]
            if len(pred.norm_ids) != 0 and pred.norm_ids[0] in gold.norm_ids:
                ct_correct += 1


    if ct_gold == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_predicted * 1.0 / ct_gold
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure

def process_one_doc(doc, entities, dict, vsm_model):

    for entity in entities:
        with torch.no_grad():
            tokens_id = vsm_model.batch_name_to_ids(entity.name)

            values, indices = vsm_model.forward(tokens_id)

            norm_id = vsm_model.get_dict_name(indices.item())
            name = dict[norm_id]
            entity.norm_ids.append(norm_id)
            entity.norm_names.append(name)
            entity.norm_confidences.append(values.item())
